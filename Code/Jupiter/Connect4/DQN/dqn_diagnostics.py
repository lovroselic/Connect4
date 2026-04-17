# DQN/dqn_diagnostics.py
from __future__ import annotations

import torch
import numpy as np

from C4.CNet192 import CNet192

COLS = 7
NEG_INF = -1e9


# --- DQN views of CNet192 ---------------------------------------------------

class CNet192_Adv(CNet192):
    """
    Returns only the policy head logits (Advantage-like).
    Useful for debugging, but NOT a calibrated Q-function by itself.
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, _v = super().forward(x)
        return a  # (B,7)


class CNet192_DuelingQ(CNet192):
    """
    Proper dueling output:
      Q = V + (A - mean(A))
    where:
      A = policy head logits (B,7)
      V = value head scalar (B,)
    """
    def __init__(self, *args, ignore_loaded_value: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.ignore_loaded_value = bool(ignore_loaded_value)

    def reset_value_head_to_zero(self):
        # makes V(s)=0 initially, then learns from TD (often stabilizes transfer)
        with torch.no_grad():
            self.value_out.weight.zero_()
            self.value_out.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, v = super().forward(x)  # a:(B,7), v:(B,)
        if self.ignore_loaded_value:
            v = torch.zeros_like(v)
        q = v.unsqueeze(1) + (a - a.mean(dim=1, keepdim=True))
        return q  # (B,7)


def freeze_cnet192_conv_block(m: CNet192, freeze: bool = True) -> None:
    convs = [m.conv1, m.conv2]
    if getattr(m, "conv_mid", None) is not None:
        convs.append(m.conv_mid)
    for layer in convs:
        for p in layer.parameters():
            p.requires_grad = not freeze


# --- state/Q helpers --------------------------------------------------------

def _ensure_bchw(state: np.ndarray) -> np.ndarray:
    """
    Accepts common Connect4 state shapes and returns (B,C,H,W).

    Your env returns (1,6,7) for get_state(). :contentReference[oaicite:2]{index=2}
    """
    if not isinstance(state, np.ndarray):
        state = np.asarray(state)

    if state.ndim == 3 and state.shape[0] == 1 and state.shape[1:] == (6, 7):
        # (B,H,W) -> (B,1,H,W)
        return state[:, None, :, :]

    if state.ndim == 4:
        # assume already BCHW
        if state.shape[1] in (1, 2, 4) and state.shape[2:] == (6, 7):
            return state
        # maybe NHWC
        if state.shape[-1] in (1, 2, 4) and state.shape[1:3] == (6, 7):
            return np.transpose(state, (0, 3, 1, 2))

    raise ValueError(f"Unsupported state shape: {state.shape}")


@torch.no_grad()
def q_values_from_state(model: torch.nn.Module, state: np.ndarray, device: torch.device) -> np.ndarray:
    """
    Returns q[7] for a single state.
    Supports models that return:
      - Tensor (B,7)
      - (Tensor(B,7), something)  [legacy]
    """
    x = _ensure_bchw(state).astype(np.float32, copy=False)
    xt = torch.from_numpy(x).to(device)

    out = model(xt)
    q = out[0] if isinstance(out, (tuple, list)) else out
    q = q.squeeze(0).detach().float().cpu().numpy()
    if q.shape != (COLS,):
        raise ValueError(f"Expected q shape (7,), got {q.shape}")
    return q


def masked_argmax(q: np.ndarray, legal: list[int]) -> int:
    if not legal:
        return 0
    m = np.full(COLS, NEG_INF, dtype=np.float32)
    m[legal] = 0.0
    return int(np.argmax(q.astype(np.float32) + m))


# --- diagnostics ------------------------------------------------------------

def greedy_action_histogram(
    model: torch.nn.Module,
    env,
    device: torch.device,
    n_states: int = 256,
    max_ply: int = 18,
    seed: int = 666,
) -> np.ndarray:
    """
    Samples random playout states, then counts greedy argmax actions from the model.
    Returns counts[7].

    Requires env API:
      - reset()
      - available_actions() -> list[int]
      - step(a) -> (state, reward, done)
      - get_state(perspective=env.current_player) -> np.ndarray
      - current_player attr
    """
    rng = np.random.default_rng(int(seed))
    counts = np.zeros(COLS, dtype=np.int64)

    was_training = bool(getattr(model, "training", False))
    model.eval()

    for _ in range(int(n_states)):
        env.reset()

        steps = int(rng.integers(0, int(max_ply) + 1))
        for _t in range(steps):
            legal = env.available_actions()
            if not legal:
                break
            a = int(rng.choice(legal))
            _s, _r, done = env.step(a)
            if done:
                break

        state = env.get_state(perspective=env.current_player)
        legal = env.available_actions()

        q = q_values_from_state(model, state, device=device)
        a_star = masked_argmax(q, legal)
        counts[a_star] += 1

    if was_training:
        model.train(True)

    return counts


@torch.no_grad()
def empty_board_q_values(model: torch.nn.Module, env, device: torch.device) -> np.ndarray:
    env.reset()
    s0 = env.get_state(perspective=env.current_player)
    return q_values_from_state(model, s0, device=device)


def plot_greedy_action_histogram(counts: np.ndarray, model_name: str = "q_model"):
    import matplotlib.pyplot as plt
    counts = np.asarray(counts, dtype=np.int64)
    fig = plt.figure(figsize=(6.2, 3.2))
    ax = fig.add_subplot(111)
    ax.bar(range(COLS), counts)
    ax.set_title(f"Greedy action histogram (n={int(counts.sum())}) | {model_name}")
    ax.set_xlabel("Column")
    ax.set_ylabel("Count")
    ax.set_xticks(range(COLS))
    fig.tight_layout()
    return fig


def plot_empty_board_q(q0: np.ndarray, model_name: str = "q_model", center: bool = True):
    """
    If center=True, plots (q - mean(q)) which is often the more honest "pre-DQN transfer" view.
    """
    import matplotlib.pyplot as plt
    q0 = np.asarray(q0, dtype=np.float32)
    y = (q0 - float(q0.mean())) if center else q0

    fig = plt.figure(figsize=(6.2, 3.2))
    ax = fig.add_subplot(111)
    ax.bar(range(COLS), y)
    suffix = " (centered)" if center else ""
    ax.set_title(f"Empty board Q-values{suffix} | {model_name}")
    ax.set_xlabel("Column")
    ax.set_ylabel("Q")
    ax.set_xticks(range(COLS))
    fig.tight_layout()
    return fig


def run_quick_diagnostics(
    model: torch.nn.Module,
    env,
    device: torch.device,
    model_name: str = "q_model",
    n_states: int = 256,
    max_ply: int = 18,
    seed: int = 666,
):
    counts = greedy_action_histogram(model, env, device=device, n_states=n_states, max_ply=max_ply, seed=seed)
    q0 = empty_board_q_values(model, env, device=device)
    fig1 = plot_greedy_action_histogram(counts, model_name=model_name)
    fig2 = plot_empty_board_q(q0, model_name=model_name, center=True)
    fig3 = plot_empty_board_q(q0, model_name=model_name, center=False)
    return counts, q0, fig1, fig2, fig3
