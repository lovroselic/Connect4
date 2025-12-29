# PPO/ppo_loop_helpers.py
from __future__ import annotations
from typing import Optional, List, Callable, Dict, Any, Tuple
import numpy as np
import torch

from C4.connect4_env import Connect4Env
from PPO.ppo_utilities import select_opponent_action
from DQN.dqn_utilities import record_first_move, evaluate_final_result, map_final_result_to_reward


def encode_single_channel_agent_centric(board: np.ndarray, player: int) -> np.ndarray:
    """
    Single-channel, agent-centric encoding.

    Assumes env.board contains {-1, 0, +1}. We multiply by `player` so the current
    mover's stones become +1 and the opponent's stones become -1.

    Returns shape: (1, 6, 7) float32  (channel-first)
    """
    b = np.asarray(board, dtype=np.int8)
    enc = (b * int(player)).astype(np.float32)
    return enc[None, :, :]   # (1, H, W)


def _to_float(x) -> float:
    if x is None:
        return 0.0
    if isinstance(x, (float, int, np.floating, np.integer)):
        return float(x)
    if torch.is_tensor(x):
        return float(x.detach().item())
    try:
        return float(x)
    except Exception:
        return 0.0


def make_recompute_lp_val(
    policy,
    temperature_ref: Dict[str, float],
) -> Callable[[np.ndarray, int, np.ndarray, int], Tuple[torch.Tensor, torch.Tensor]]:

    @torch.inference_mode()
    def _recompute(state_np: np.ndarray, action_int: int, legal_mask_np: np.ndarray, ply_idx: int = -1):
        dev = next(policy.parameters()).device

        x = np.asarray(state_np, dtype=np.float32)
        s = torch.from_numpy(x).unsqueeze(0).to(dev)   # (B, C, H, W) if x is (C,H,W)

        lm_np = np.asarray(legal_mask_np, dtype=bool)
        lm = torch.from_numpy(lm_np).unsqueeze(0).to(dev)

        a = torch.tensor([int(action_int)], dtype=torch.long, device=dev)
        ply = None if ply_idx is None or int(ply_idx) < 0 else torch.tensor([int(ply_idx)], dtype=torch.long, device=dev)

        T = max(float(temperature_ref.get("T", 1.0)), 1e-6)

        lp, _, v = policy.evaluate_actions(
            s, a,
            legal_action_masks=lm,
            ply_indices=ply,
            temperature=T,
        )
        return lp[0].detach(), v[0].detach()

    return _recompute


def ppo_agent_step(
    env: Connect4Env,
    policy,
    buffer,
    temperature: float,
    recompute_fn,
    last_agent_reward_entries_ref,
    ply_idx: int,
    opening_kpis,
    openings,
):
    """
    Agent (+1) turn.
    Returns: (done, reward, new_ply_idx)
    """
    ply_used = int(ply_idx)

    legal = env.available_actions()
    if not legal:
        return True, 0.0, ply_idx

    legal = [int(a) for a in legal]
    legal_mask = np.zeros(7, dtype=bool)
    legal_mask[legal] = True

    # --- SINGLE-CHANNEL agent-centric encoding ---
    enc = encode_single_channel_agent_centric(env.board, +1)

    action, logprob, value, _info = policy.act(
        enc,
        legal,
        temperature=float(temperature),
        ply_idx=int(ply_idx),
    )

    env.current_player = +1
    _, reward, done = env.step(int(action))
    ply_idx += 1

    # opening stats (agent started)
    if ply_idx == 1:
        record_first_move(opening_kpis, int(action), who="agent")
        if openings is not None:
            openings.on_first_move(int(action), is_agent=True)

    last_agent_reward_entries_ref[0] = buffer.add_augmented(
        state_np=enc,
        action=int(action),
        logprob=_to_float(logprob),
        value=_to_float(value),
        reward=_to_float(reward),
        done=bool(done),
        legal_mask_bool=legal_mask,
        hflip=True,
        colorswap=True,
        include_original=True,
        recompute_fn=recompute_fn,
        ply_idx=ply_used,
    )

    return bool(done), float(reward), int(ply_idx)


def ppo_opponent_step(
    env: Connect4Env,
    policy,
    lookahead_mode: Optional[int | str],  # int depth or "self" or "R"/None
    temperature: float,
    buffer,
    last_agent_reward_entries_ref: List[Optional[Any]],
    loss_penalty: float,
    ply_idx: int,
    opening_kpis: Optional[Dict[str, Any]] = None,
    openings: Optional[Any] = None,
    attr_loss_to_last: bool = True,
) -> tuple[bool, int, float]:
    """
    Opponent (-1) turn.
    Returns: (done, new_ply_idx, penalty_to_agent)
    """
    legal = env.available_actions()
    if not legal:
        return True, ply_idx, 0.0

    legal = [int(a) for a in legal]

    mode = lookahead_mode
    if mode == "POP":
        mode = "self"

    # Choose opponent action
    if mode in (None, "R", "rand", "Random", 0):
        opp_action = int(np.random.choice(legal))

    elif mode == "self":
        # --- SINGLE-CHANNEL opponent-centric encoding (player=-1 => opponent stones become +1) ---
        enc_opp = encode_single_channel_agent_centric(env.board, -1)
        opp_action, _, _, _ = policy.act(
            enc_opp,
            legal,
            temperature=float(temperature),
            ply_idx=int(ply_idx),
        )
        opp_action = int(opp_action)

    else:
        depth = int(mode)
        opp_action = int(select_opponent_action(env.board.copy(), player=-1, depth=depth))

    env.current_player = -1
    _, oppo_reward, done = env.step(int(opp_action))  # mover-centric (opponent POV)

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
        agent_penalty = -float(oppo_reward) * OPPO_PENALTY_FACTOR

        # Only force LOSS_PENALTY if opponent actually won (not draw)
        if done and getattr(env, "winner", None) == -1:
            agent_penalty = float(loss_penalty)

        if agent_penalty != 0.0:
            buffer.add_penalty_to_entries(
                last_agent_reward_entries_ref[0],
                float(agent_penalty),
                scale_raw=True,
            )
            penalty_to_agent = float(agent_penalty)

    return bool(done), int(ply_idx + 1), float(penalty_to_agent)


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
    Returns: (done, new_ply_idx, penalty)
    """
    ply_idx = 0
    penalty = 0.0
    if episode % 2 == 1:
        done, ply_idx, penalty = ppo_opponent_step(
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
        return done, ply_idx, penalty

    return False, ply_idx, penalty


def ppo_finalize_if_done(env: Connect4Env) -> tuple[bool, Optional[float], float]:
    if env.done:
        final_result = evaluate_final_result(env, agent_player=+1)
        total_reward = map_final_result_to_reward(final_result)
        return True, final_result, total_reward
    return False, None, 0.0
