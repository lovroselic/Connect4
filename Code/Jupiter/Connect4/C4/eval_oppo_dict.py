# C4.eval_oppo_dict.py
# 
import numpy as np 

#final evaluation
EVALUATION_OPPONENTS = {
    "Random": 100,
    "Leftmost": 100,
    "Center": 100,
    "Lookahead-1": 100,
    "Lookahead-2": 100,
    "Lookahead-3": 100,
    "Lookahead-4": 100,
    "Lookahead-5": 50,
    "Lookahead-6": 24,
    "Lookahead-7": 10,
    "Lookahead-9": 6,
    "Lookahead-11": 4,
    "Lookahead-13": 4,
   
}

#online training evals
EVAL_CFG = {
    "Random": 100, 
    "Leftmost": 100,
    "Center": 100,
    "Lookahead-1": 100,
    "Lookahead-2": 10,
    "Lookahead-3": 30,
    "Lookahead-4": 10,
    "Lookahead-5": 10,
    "Lookahead-6": 4,
    "Lookahead-7": 4,
    "Lookahead-9": 4,
    "Lookahead-11": 2,
    "Lookahead-13": 2,
    } 

#OPENING_NOISE_K = {0:0.85, 1:0.10, 2:0.05}
OPENING_NOISE_K = {0:0.97, 1:0.02, 2:0.01}

def sample_opening_noise_k(opening_noise_cfg, rng) -> int:
    """
    opening_noise_cfg can be:
      - int: returned as-is
      - dict {k:int -> prob:float}: sampled
      - list/tuple of ints: uniform sample
    """
    if opening_noise_cfg is None:
        return 0
    if isinstance(opening_noise_cfg, (int, np.integer)):
        return int(opening_noise_cfg)
    if isinstance(opening_noise_cfg, dict):
        ks = np.array(list(opening_noise_cfg.keys()), dtype=np.int64)
        ps = np.array([float(opening_noise_cfg[k]) for k in ks], dtype=np.float64)
        ps = ps / ps.sum() if ps.sum() > 0 else np.ones_like(ps) / len(ps)
        return int(rng.choice(ks, p=ps))
    if isinstance(opening_noise_cfg, (list, tuple, np.ndarray)):
        return int(rng.choice(np.asarray(opening_noise_cfg, dtype=np.int64)))
    return int(opening_noise_cfg)  # last-ditch
