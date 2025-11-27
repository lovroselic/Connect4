# PPO/ppo_loop_helpers.py
from __future__ import annotations
from typing import Optional, List, Callable, Dict, Any, Tuple
import numpy as np
import torch

from C4.connect4_env import Connect4Env
from PPO.ppo_utilities import encode_two_channel_agent_centric, select_opponent_action, encode_four_channel_agent_centric
from DQN.dqn_utilities import record_first_move, evaluate_final_result, map_final_result_to_reward


NEG_INF = -1e9


def make_recompute_lp_val(
    policy,
    device: torch.device,
    temperature_ref: Dict[str, float],
) -> Callable[[np.ndarray, int, np.ndarray], Tuple[torch.Tensor, torch.Tensor]]:
    """
    Build a callable:
        recompute_fn(state_np, action_int, legal_mask_np) -> (logprob, value)

    where:
      - state_np is the same shape as stored in the buffer, e.g.
            (2,6,7) agent-centric planes [me, opp]
         or (4,6,7) [me, opp, row, col]
         or (6,7) scalar board in {-1,0,+1}
      - legal_mask_np is shape (A,) bool, A == policy.action_dim
      - action_int is in [0, A-1]

    The function:
      - runs a fresh forward pass through `policy`
      - applies the legal-action mask
      - uses the CURRENT temperature in temperature_ref["T"]
      - returns (logprob_of_given_action, value)
    """

    @torch.inference_mode()
    def _recompute(state_np: np.ndarray, action_int: int, legal_mask_np: np.ndarray):
        # --- state to tensor; shape becomes [1, ..., 6, 7] ---
        x = np.asarray(state_np, dtype=np.float32)
        s = torch.from_numpy(x).unsqueeze(0).to(device)  # (1,C,6,7) or (1,6,7)

        # Forward through ActorCritic (handles 1/2/4/6 channels internally)
        logits, v = policy.forward(s)  # logits: [1,A], v: [1] or [1,1]

        # --- legal mask ---
        lm_np = np.asarray(legal_mask_np, dtype=bool)
        lm = torch.from_numpy(lm_np).unsqueeze(0).to(device=device)  # (1,A)
        if lm.sum().item() == 0:
            # Safety: ensure at least something is legal
            lm = torch.ones_like(lm, dtype=torch.bool)

        masked = logits.masked_fill(~lm, NEG_INF)

        # --- temperature ---
        T = float(temperature_ref.get("T", 1.0))
        if T <= 0.0:
            # degenerate "argmax-style" – but still use log_softmax for numeric sanity
            log_probs = torch.log_softmax(masked, dim=-1)
        else:
            log_probs = torch.log_softmax(masked / T, dim=-1)

        a_idx = int(action_int)
        lp = log_probs[0, a_idx]
        return lp.detach(), v.squeeze(0).detach()

    return _recompute


def ppo_agent_step(
    env: Connect4Env,
    policy,
    buffer,
    temperature: float,
    recompute_fn,  # still used for aug recompute
    last_agent_reward_entries_ref,
    ply_idx: int,
    opening_kpis,
    openings,
):
    """
    Agent (+1) turn. Uses policy.act with built-in heuristics (win_now/center/guard).
    """
    legal = env.available_actions()
    if not legal: return True, 0.0, ply_idx

    legal = [int(a) for a in legal]
    legal_mask = np.zeros(7, dtype=bool)
    legal_mask[legal] = True

    # agent-centric encoding
    enc = encode_four_channel_agent_centric(env.board, +1)

    # policy decides (with heuristics)
    action, logprob, value, info = policy.act(enc,legal,temperature=temperature, ply_idx=ply_idx)

    env.current_player = +1
    _, reward, done = env.step(action)
    ply_idx += 1

    # opening stats (agent started)
    if ply_idx == 1:
        record_first_move(opening_kpis, int(action), who="agent")
        if openings is not None:
            openings.on_first_move(int(action), is_agent=True)

    # write augmented samples
    last_agent_reward_entries_ref[0] = buffer.add_augmented(
        state_np=enc,
        action=int(action),
        logprob=logprob,
        value=value,
        reward=reward,
        done=done,
        legal_mask_bool=legal_mask,
        hflip=True,
        colorswap=False,
        include_original=True,
        recompute_fn=recompute_fn,
    )

    return done, float(reward), ply_idx


def ppo_opponent_step(
    env: Connect4Env,
    policy,
    lookahead_mode: Optional[int | str],  # int depth or "self" or None
    temperature: float,
    buffer,
    last_agent_reward_entries_ref: List[Optional[Any]],
    loss_penalty: float,              # <- can now be unused
    ply_idx: int,
    opening_kpis: Optional[Dict[str, Any]] = None,
    openings: Optional[Any] = None,
    attr_loss_to_last: bool = True,
) -> tuple[bool, int, float]:
    """
    Opponent (-1) turn.

    Returns:
        done (bool),
        new_ply_idx (int),
        penalty_to_agent (float)  # for ep_return logging
    """
    legal = env.available_actions()
    if not legal: return True, ply_idx, 0.0

    # Choose opponent action
    if lookahead_mode == "self":
        enc_opp = encode_two_channel_agent_centric(env.board, -1)
        opp_action, _, _, _ = policy.act(enc_opp, legal, temperature=temperature)
    else:
        opp_action = select_opponent_action(env.board.copy(), player=-1, depth=lookahead_mode)

    env.current_player = -1
    _, oppo_reward, done = env.step(opp_action)   # <- mover-centric, from opp POV

    # First-move stats (opponent starts)
    if ply_idx == 0:
        if opening_kpis is not None:
            record_first_move(opening_kpis, int(opp_action), who="opp")
        if openings is not None:
            openings.on_first_move(int(opp_action), is_agent=False)

    penalty_to_agent = 0.0
    
    OPPO_PENALTY_FACTOR = 1.0

    # Convert opponent reward into agent-centric penalty
    if attr_loss_to_last and last_agent_reward_entries_ref[0] is not None:
        # FULL zero-sum: agent sees negative of opponent reward
        agent_penalty = -float(oppo_reward) * OPPO_PENALTY_FACTOR 
        if done: agent_penalty = env.LOSS_PENALTY   #if we use penalty factor, this one is not factored!
        if agent_penalty != 0.0:
            buffer.add_penalty_to_entries(
                last_agent_reward_entries_ref[0],
                agent_penalty,
                scale_raw=True,
            )
            penalty_to_agent = agent_penalty

    return done, ply_idx + 1, penalty_to_agent


def ppo_maybe_opponent_opening(
    env: Connect4Env,
    policy,
    episode: int,
    lookahead_mode: Optional[int | str],
    temperature: float,
    buffer,
    last_agent_reward_entries_ref: List[Optional[Any]],
    loss_penalty: float,
    opening_kpis: Optional[Dict[str, Any]] = None,
    openings: Optional[Any] = None,
    attr_loss_to_last: bool = True,
) -> tuple[bool, int, float]:
    """
    If the opponent should start (odd episodes), play exactly one opponent turn.

    Returns:
        done (bool), new_ply_idx (int)
    """
    ply_idx = 0
    oppo_reward = 0
    if episode % 2 == 1:
        done, ply_idx, oppo_reward = ppo_opponent_step(
            env=env,
            policy=policy,
            lookahead_mode=lookahead_mode,
            temperature=temperature,
            buffer=buffer,
            last_agent_reward_entries_ref=last_agent_reward_entries_ref,
            loss_penalty=loss_penalty,
            ply_idx=ply_idx,
            opening_kpis=opening_kpis,
            openings=openings,
            attr_loss_to_last=attr_loss_to_last,
        )
        return done, ply_idx, oppo_reward

    # Agent starts; no moves yet
    return False, ply_idx, oppo_reward


def ppo_finalize_if_done(env: Connect4Env) -> tuple[bool, Optional[float], float]:
    """
    DQN-style finalization helper.

    Returns:
        is_done (bool),
        final_result (1 win, -1 loss, 0.5 draw, or None),
        total_reward (mapped WIN/DRAW/LOSS reward)
    """
    if env.done:
        final_result = evaluate_final_result(env, agent_player=+1)
        total_reward = map_final_result_to_reward(final_result)
        return True, final_result, total_reward
    return False, None, 0.0
