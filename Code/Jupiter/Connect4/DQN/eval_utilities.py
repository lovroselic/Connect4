# DQN/eval_utilities.py

import numpy as np
import torch

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
