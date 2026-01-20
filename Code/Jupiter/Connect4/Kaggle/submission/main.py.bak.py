# kaggle: main.py
# tar -czf submit.tar.gz -C submission main.py PPO_14.pt

import os
from typing import Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_DEVICE = torch.device("cpu")
_MODEL: Optional[nn.Module] = None
_MODEL_LOAD_FAILED: bool = False

_CENTER_COL = 3
MODEL = "PPO_14.pt"


# ---------- CNet192 ----------
class CNet192(nn.Module):
    """
    CNet192:
      - Conv( in_ch -> 192, k=4, pad=0): 6x7 -> 3x4
      - Optional mid Conv(192 -> 192, k=3, pad=1): 3x4 -> 3x4
      - Conv(192 -> 192, k=2, pad=0): 3x4 -> 2x3
      - FC to 192
      - Policy head: 192 -> 192 -> 7
      - Value head : 192 -> 192 -> 1
    """
    def __init__(self, in_channels: int = 1, use_mid_3x3: bool = True):
        super().__init__()
        self.in_channels = int(in_channels)
        self.use_mid_3x3 = bool(use_mid_3x3)

        self.conv1 = nn.Conv2d(self.in_channels, 192, kernel_size=4, padding=0)
        self.conv_mid = nn.Conv2d(192, 192, kernel_size=3, padding=1) if self.use_mid_3x3 else None
        self.conv2 = nn.Conv2d(192, 192, kernel_size=2, padding=0)

        # fixed for this architecture
        self.flat = 192 * 2 * 3

        self.fc = nn.Linear(self.flat, 192)
        self.policy_fc = nn.Linear(192, 192)
        self.policy_out = nn.Linear(192, 7)

        self.value_fc = nn.Linear(192, 192)
        self.value_out = nn.Linear(192, 1)

    def _forward_conv(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        if self.conv_mid is not None:
            x = F.relu(self.conv_mid(x))
        x = F.relu(self.conv2(x))
        return x

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self._forward_conv(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))

        pol = F.relu(self.policy_fc(x))
        pol = self.policy_out(pol)            # (B,7)

        val = F.relu(self.value_fc(x))
        val = self.value_out(val).squeeze(-1) # (B,)

        return pol, val


def _torch_load_any(path: str):
    # PyTorch versions differ; weights_only may exist or not
    try:
        return torch.load(path, map_location=_DEVICE, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=_DEVICE)


def _strip_prefix_if_all(sd: dict, prefix: str) -> dict:
    keys = list(sd.keys())
    if keys and all(isinstance(k, str) and k.startswith(prefix) for k in keys):
        return {k[len(prefix):]: v for k, v in sd.items()}
    return sd


def _extract_state_and_use_mid(ckpt):
    """
    Accept:
      - save_cnet192 payload: {'cfg':..., 'model_state_dict': ...}
      - {'state_dict': ...}
      - raw state_dict
      - ActorCritic state_dict with 'net.' prefix
      - DataParallel with 'module.' prefix
    Returns: (state_dict, use_mid_3x3)
    """
    cfg = {}
    state = None

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt and isinstance(ckpt["model_state_dict"], dict):
        cfg = dict(ckpt.get("cfg", {}) or {})
        state = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        cfg = dict(ckpt.get("cfg", {}) or {})
        state = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and ckpt and all(torch.is_tensor(v) for v in ckpt.values()):
        state = ckpt  # raw state_dict
    else:
        raise RuntimeError("Unrecognized checkpoint format")

    # unwrap common wrappers safely
    state = _strip_prefix_if_all(state, "module.")
    state = _strip_prefix_if_all(state, "net.")

    # decide mid-conv from cfg if present, otherwise infer from keys
    use_mid = bool(cfg.get("use_mid_3x3", any(k.startswith("conv_mid.") for k in state.keys())))

    return state, use_mid


def _load_model_once() -> Optional[nn.Module]:
    global _MODEL, _MODEL_LOAD_FAILED
    if _MODEL is not None:
        return _MODEL
    if _MODEL_LOAD_FAILED:
        return None

    # model is in same dir as main.py, but __file__ can be flaky in some wrappers, so try multiple
    candidates = []
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        candidates.append(os.path.join(here, MODEL))
    except Exception:
        pass
    candidates.append(os.path.join(os.getcwd(), MODEL))
    candidates.append(os.path.join("/kaggle_simulations/agent", MODEL))

    ckpt_path = None
    for p in candidates:
        if os.path.exists(p):
            ckpt_path = p
            break

    if ckpt_path is None:
        _MODEL_LOAD_FAILED = True
        return None

    try:
        ckpt = _torch_load_any(ckpt_path)
        state, use_mid = _extract_state_and_use_mid(ckpt)

        m = CNet192(in_channels=1, use_mid_3x3=use_mid).to(_DEVICE)
        m.load_state_dict(state, strict=True)
        m.eval()

        _MODEL = m
        return _MODEL
    except Exception:
        _MODEL_LOAD_FAILED = True
        return None


def _fallback_move(grid: np.ndarray) -> int:
    # center if possible else closest-to-center legal
    if grid[0, 3] == 0:
        return 3
    legal = [c for c in range(7) if grid[0, c] == 0]
    if not legal:
        return 0
    return int(sorted(legal, key=lambda c: (abs(c - 3), c))[0])


def _argmax_legal_center_tiebreak(vals: np.ndarray, legal: List[int], center: int = 3, tol: float = 1e-12) -> int:
    best = max(vals[c] for c in legal)
    tied = [c for c in legal if abs(vals[c] - best) <= tol]
    if len(tied) == 1:
        return int(tied[0])
    return int(sorted(tied, key=lambda c: (abs(c - center), c))[0])


# ---------- Kaggle agent ----------
def agent(obs, config):
    try:
        mark = int(obs["mark"]) if isinstance(obs, dict) else int(obs.mark)
        flat = obs["board"] if isinstance(obs, dict) else obs.board
        step = int(obs.get("step", 0)) if isinstance(obs, dict) else int(getattr(obs, "step", 0))

        grid = np.asarray(flat, dtype=np.int8).reshape(6, 7)

        # Use the big first-step budget to load
        if step == 0:
            _load_model_once()

        legal = [c for c in range(7) if grid[0, c] == 0]
        if not legal:
            return 0

        stones = int(np.count_nonzero(grid))
        if stones == 0:
            # opening book
            return 3

        model = _load_model_once()
        if model is None:
            return _fallback_move(grid)

        # POV scalar: current player = +1, opponent = -1
        pov = np.zeros_like(grid, dtype=np.float32)
        pov[grid == mark] = 1.0
        pov[(grid != 0) & (grid != mark)] = -1.0

        x = torch.from_numpy(pov).unsqueeze(0).unsqueeze(0).to(_DEVICE)

        with torch.inference_mode():
            logits, _ = model(x)
            logits = logits[0].detach().cpu().numpy().astype(np.float64)

        # mask illegal -> -inf
        masked = np.full(7, -1e18, dtype=np.float64)
        for c in legal:
            masked[c] = logits[c]

        return int(_argmax_legal_center_tiebreak(masked, legal, center=_CENTER_COL))

    except Exception:
        # Never crash validation
        try:
            flat = obs["board"] if isinstance(obs, dict) else obs.board
            grid = np.asarray(flat, dtype=np.int8).reshape(6, 7)
            return _fallback_move(grid)
        except Exception:
            return 0
