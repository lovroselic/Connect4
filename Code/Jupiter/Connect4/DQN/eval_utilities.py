# DQN/eval_utilities.py

import numpy as np
import torch
import random

def try_agent_preprocess(agent, state_np, device):
    """
    Try to use the agent's own preprocessing if available.
    Must return a (1, C, 6, 7) float tensor on the correct device.
    """
    for name in ("state_to_tensor", "_state_to_tensor", "encode_state", "_encode_state"):
        fn = getattr(agent, name, None)
        if callable(fn):
            # handle both (state) and (state, player) signatures
            try:
                x = fn(state_np) if fn.__code__.co_argcount == 2 else fn(state_np, player=+1)
            except Exception:
                x = fn(state_np)  # fall back gracefully

            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x)

            if x.dim() == 3:         # (C, 6, 7)
                x = x.unsqueeze(0)   # (1, C, 6, 7)
            elif x.dim() == 2:       # (6, 7) – unlikely if using agent fn, but guard it
                x = x.unsqueeze(0).unsqueeze(0)

            return x.to(device).float()
    return None

def two_channel_absolute(state_np, device):
    """
    Fallback encoder for boards with values {+1, -1, 0}.
    Returns (1, 2, 6, 7) float tensor: [agent(+1) plane, opp(-1) plane].
    """
    s = np.array(state_np, dtype=np.float32)
    p_agent = (s == +1).astype(np.float32)
    p_oppo  = (s == -1).astype(np.float32)
    stacked = np.stack([p_agent, p_oppo], axis=0)  # (2, 6, 7)
    return torch.from_numpy(stacked).unsqueeze(0).to(device).float()

def build_input(agent, state_np, device):
    """
    Returns (1, C, 6, 7) float tensor on device.
    Prefers agent's own preprocessing; falls back to two-channel absolute.
    """
    x = try_agent_preprocess(agent, state_np, device)
    if x is None:
        x = two_channel_absolute(state_np, device)
    return x


def _valid_actions_of(env):
    # support either API name
    return env.available_actions() if hasattr(env, "available_actions") else env.valid_actions()

def _mask_argmax(q, valid_actions):
    q = np.asarray(q, dtype=np.float32)
    mask = np.full_like(q, -1e9)
    for a in valid_actions: mask[a] = 0.0
    a = int(np.argmax(q + mask))
    return a if a in valid_actions else random.choice(valid_actions)

@torch.no_grad()
def evaluate_with_leaks(
    agent,
    env,
    episodes=200,
    opponent="lookahead",   # "random" or "lookahead"
    depth=1,                # opponent lookahead depth
    use_guard=True,         # force guard behavior on exploit in eval
    epsilon_override=0.0,   # 0 => pure exploit
    device=None,
):
    """Run evaluation with ε=0 (by default), no replay writes, and track 1‑ply leaks."""
    device = device or agent.device

    # ---- save & override exploration / guard flags for eval ----
    old_eps = agent.epsilon
    agent.epsilon = float(epsilon_override)
    old_guard_prob = getattr(agent, "guard_prob", 0.0)
    if use_guard:
        agent.guard_prob = 1.0  # guard always on during eval

    wins = losses = draws = 0
    gave_opp_win = 0           # chose a move that lets opp win next
    missed_our_win = 0         # had a 1‑ply win but didn't take it
    agent_moves = 0

    LA = getattr(agent, "lookahead", None)

    for _ in range(episodes):
        # reset
        rs = env.reset()
        state = rs[0] if isinstance(rs, (tuple, list)) else rs
        done = False

        while not done:
            valid_actions = _valid_actions_of(env)

            if env.current_player == 1:
                # ----- AGENT TURN -----
                has_win, winning_a = agent.had_one_ply_win(np.array(state), valid_actions, +1)

                # pure model forward (ε already set to 0 by default)
                x = torch.tensor(
                    agent._agent_planes(np.array(state), agent_side=+1),
                    dtype=torch.float32, device=device
                ).unsqueeze(0)
                q = agent.model(x).squeeze(0).cpu().numpy()
                action = _mask_argmax(q, valid_actions)

                # leak accounting
                if agent.gave_one_ply_win(np.array(state), action, +1):
                    gave_opp_win += 1
                if has_win and action != winning_a:
                    missed_our_win += 1
                agent_moves += 1

            else:
                # ----- OPPONENT TURN -----
                if opponent == "random":
                    action = random.choice(valid_actions)
                else:
                    a = None
                    if LA is not None:
                        a = LA.n_step_lookahead(np.array(state), player=-1, depth=int(depth))
                    action = a if (a in valid_actions) else random.choice(valid_actions)

            # step (support 3‑ or 4‑tuple envs)
            step_ret = env.step(action)
            if len(step_ret) == 3:
                state, reward, done = step_ret
                info = {}
            else:
                state, reward, done, info = step_ret

        # ---- episode outcome ----
        winner = None
        if hasattr(env, "winner"):
            winner = env.winner
        elif hasattr(env, "get_winner"):
            winner = env.get_winner()
        elif "winner" in info:
            winner = info["winner"]
        else:
            winner = int(np.sign(agent._is_win(np.array(state))))

        if winner == +1: wins += 1
        elif winner == -1: losses += 1
        else: draws += 1

    # ---- restore knobs ----
    agent.epsilon = old_eps
    agent.guard_prob = old_guard_prob

    # ---- report ----
    leak_den = max(1, agent_moves)
    return {
        "episodes": episodes,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": wins / episodes,
        "gave_opp_win_rate": gave_opp_win / leak_den,
        "missed_our_win_rate": missed_our_win / leak_den,
        "agent_moves": agent_moves,
        "opponent": opponent,
        "depth": depth,
    }


