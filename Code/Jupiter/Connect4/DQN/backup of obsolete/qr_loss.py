# DQN/qr_loss.py
import torch

@torch.no_grad()
def build_taus(n_quantiles: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """taus in (0,1): (i-0.5)/N, shape [N]"""
    taus = (torch.arange(n_quantiles, device=device, dtype=dtype) + 0.5) / float(n_quantiles)
    return taus

def quantile_huber_loss(pred: torch.Tensor,
                        target: torch.Tensor,
                        taus: torch.Tensor,
                        kappa: float = 1.0,
                        isw: torch.Tensor | None = None) -> torch.Tensor:
    """
    pred:   [B, N]  predicted quantiles for chosen actions
    target: [B, N]  target quantiles for greedy next action
    taus:   [N]     quantile fractions (0,1)
    kappa:  Huber threshold
    isw:    [B] optional importance-sampling weights (PER)

    Returns scalar loss.
    """
    B, N = pred.shape
    # Pairwise TD errors u_{ij} = tgt_j - pred_i  -> shape [B, N_pred, N_tgt]
    u = target.unsqueeze(1) - pred.unsqueeze(2)

    abs_u = u.abs()
    huber = torch.where(abs_u <= kappa, 0.5 * u**2, kappa * (abs_u - 0.5 * kappa))

    tau = taus.view(1, N, 1)                          # [1, N, 1]
    weight = (tau - (u.detach() < 0).float()).abs()   # |tau - 1_{u<0}|

    per_sample = (weight * huber).mean(dim=(1, 2))    # average over both quantile axes
    if isw is not None:
        per_sample = (per_sample * isw)

    return per_sample.mean()
