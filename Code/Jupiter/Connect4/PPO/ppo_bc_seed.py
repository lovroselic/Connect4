# -*- coding: utf-8 -*-
# PPO/ppo_bc_seed.py
# Behavior Cloning (supervised) warm-start for PPO ActorCritic.

from __future__ import annotations
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from PPO.actor_critic import ActorCritic, NEG_INF
from PPO.ppo_utilities import encode_four_channel_agent_centric  # 4-ch agent-centric

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------- dataset adapter --------
class MovesDataset(Dataset):
    """
    Expects iterable of samples with keys:
      - 'board':  (6,7) np.int8 in {-1,0,+1}
      - 'player': +1 or -1 (who moved)
      - 'action': int in [0..6]
    (Optionally 'legal': list[int]; if absent we infer from board.)

    Output:
      - state:      (4,6,7) float32, agent-centric [me, opp, row, col]
      - action:     int in [0..6]
      - legal_mask: (7,) bool
    """
    def __init__(self, samples):
        self.samples = list(samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        board  = np.asarray(s["board"], dtype=np.int8)   # (6,7)
        player = int(s["player"])
        action = int(s["action"])

        # infer legal if not provided
        legal = s.get("legal", None)
        if legal is None:
            # legal: columns where TOP cell is still empty
            legal = [c for c in range(7) if board[0, c] == 0]

        legal_mask = np.zeros(7, dtype=bool)
        if len(legal) == 0:  # guard: if something is off, allow all
            legal_mask[:] = True
        else:
            legal_mask[legal] = True

        # encode to 4-channel agent-centric [me, opp, row, col]
        s4 = encode_four_channel_agent_centric(board, player)  # (4,6,7)
        return s4.astype(np.float32), action, legal_mask


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
    augment_hflip: bool = True,
):
    ds = MovesDataset(samples_iterable)
    if len(ds) == 0:
        if verbose: print("[BC] No samples provided; skipping.")
        return {"loss_ce": [], "loss_ent": []}

    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)
    policy.train()
    opt = torch.optim.Adam(policy.parameters(), lr=lr, eps=1e-5)

    loss_ce_hist, loss_ent_hist = [], []
    t0 = time.time()

    A = int(policy.action_dim)  # number of columns (7)

    # ---- NEW: tqdm over epochs, with safe initial description ----
    if verbose:
        epoch_bar = tqdm(
            total=epochs,
            desc=f"[BC] Epoch 0/{epochs} | CE=---- | Ent=----",
            leave=True,
            dynamic_ncols=True,
        )
    else:
        epoch_bar = None

    for ep in range(1, epochs + 1):
        ce_sum = ent_sum = n_batches = 0

        for s4, act, legal_mask in loader:
            # Base batch: [B,4,6,7], [B], [B,7]
            s4  = s4.to(DEVICE)                            # [B,4,6,7]
            act = act.to(DEVICE, dtype=torch.long)         # [B]
            lm  = legal_mask.to(DEVICE, dtype=torch.bool)  # [B,7]

            states_list  = [s4]
            actions_list = [act]
            masks_list   = [lm]

            if augment_hflip:
                s4_h  = torch.flip(s4, dims=[-1])
                act_h = (A - 1) - act
                lm_h  = torch.flip(lm, dims=[1])

                states_list.append(s4_h)
                actions_list.append(act_h)
                masks_list.append(lm_h)

            states      = torch.cat(states_list, dim=0)
            actions     = torch.cat(actions_list, dim=0)
            legal_masks = torch.cat(masks_list, dim=0)

            logits, _ = policy.forward(states)
            logits = logits.masked_fill(~legal_masks, NEG_INF)

            loss_ce = F.cross_entropy(logits, actions)

            probs  = torch.softmax(logits, dim=-1)
            p_safe = probs.clamp_min(1e-12)
            ent    = -(p_safe * p_safe.log()).sum(dim=-1).mean()

            loss = loss_ce - ent_coef * ent

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            opt.step()

            ce_sum  += float(loss_ce.item())
            ent_sum += float(ent.item())
            n_batches += 1

        # epoch means
        mean_ce  = ce_sum  / max(1, n_batches)
        mean_ent = ent_sum / max(1, n_batches)
        loss_ce_hist.append(mean_ce)
        loss_ent_hist.append(mean_ent)

        if epoch_bar is not None:
            epoch_bar.set_description(
                f"[BC] Epoch {ep}/{epochs} | CE={mean_ce:.4f} | Ent={mean_ent:.4f}"
            )
            epoch_bar.update(1)

    if epoch_bar is not None:
        epoch_bar.close()

    if verbose:
        dt = time.time() - t0
        print(f"[BC] Done in {dt:.1f}s | Samples: {len(ds)}")

    return {"loss_ce": loss_ce_hist, "loss_ent": loss_ent_hist}
