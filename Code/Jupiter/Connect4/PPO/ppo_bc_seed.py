# -*- coding: utf-8 -*-
# PPO/ppo_bc_seed.py
# Behavior Cloning (supervised) warm-start for PPO ActorCritic.

from __future__ import annotations
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from PPO.actor_critic import ActorCritic
from PPO.ppo_utilities import encode_two_channel_agent_centric

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NEG_INF = -1e9

# -------- dataset adapter --------
class MovesDataset(Dataset):
    """
    Expects iterable of samples with keys:
      - 'board': (6,7) np.int8 in {-1,0,+1}
      - 'player': +1 or -1 (who moved)
      - 'action': int in [0..6]
    (Optionally you can pass 'legal': list[int]; if absent we infer from board.)
    """
    def __init__(self, samples):
        self.samples = list(samples)

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        board  = np.asarray(s["board"], dtype=np.int8)
        player = int(s["player"])
        action = int(s["action"])
        # infer legal if not provided
        legal = s.get("legal", None)
        if legal is None:
            legal = [c for c in range(7) if board[0, c] == 0]
        legal_mask = np.zeros(7, dtype=bool)
        if len(legal) == 0:  # guard
            legal_mask[:] = True
        else:
            legal_mask[legal] = True
        # encode to 2ch agent-centric
        s2 = encode_two_channel_agent_centric(board, player)  # (2,6,7)
        return s2.astype(np.float32), action, legal_mask

# -------- BC training --------
@torch.inference_mode(False)
def bc_pretrain_actor(
    policy: ActorCritic,
    samples_iterable,
    epochs: int = 3,
    batch_size: int = 2048,
    lr: float = 3e-4,
    ent_coef: float = 0.0,
    max_grad_norm: float = 0.5,
    verbose: bool = True,
):
    """
    Supervised cross-entropy on expert actions with illegal-action masking.
    Does NOT update the value head (optionally could, if you have outcome targets).
    """
    ds = MovesDataset(samples_iterable)
    if len(ds) == 0:
        if verbose:
            print("[BC] No samples provided; skipping.")
        return {"loss_ce": [], "loss_ent": []}

    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)
    policy.train()
    opt = torch.optim.Adam(policy.parameters(), lr=lr, eps=1e-5)

    loss_ce_hist, loss_ent_hist = [], []
    t0 = time.time()

    for ep in range(1, epochs + 1):
        ce_sum = ent_sum = n_batches = 0
        for s2, act, legal_mask in loader:
            s2 = s2.to(DEVICE)                             # [B,2,6,7]
            act = act.to(DEVICE, dtype=torch.long)         # [B]
            lm  = legal_mask.to(DEVICE, dtype=torch.bool)  # [B,7]

            logits, _ = policy.forward(s2)                 # [B,7], [B]
            # mask illegal logits
            logits = logits.masked_fill(~lm, NEG_INF)

            # cross-entropy on the expert action
            loss_ce = F.cross_entropy(logits, act)

            # optional entropy bonus (encourages broader initial policy)
            probs = F.softmax(logits, dim=-1)
            ent = -(probs.clamp_min(1e-12) * probs.clamp_min(1e-12).log()).sum(dim=-1).mean()
            loss = loss_ce - ent_coef * ent

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            opt.step()

            ce_sum  += float(loss_ce.item())
            ent_sum += float(ent.item())
            n_batches += 1

        loss_ce_hist.append(ce_sum / max(1, n_batches))
        loss_ent_hist.append(ent_sum / max(1, n_batches))
        if verbose:
            print(f"[BC] Epoch {ep}/{epochs} | CE={loss_ce_hist[-1]:.4f} | Ent={loss_ent_hist[-1]:.4f}")

    if verbose:
        print(f"[BC] Done in {time.time() - t0:.1f}s | Samples: {len(ds)}")
    return {"loss_ce": loss_ce_hist, "loss_ent": loss_ent_hist}
