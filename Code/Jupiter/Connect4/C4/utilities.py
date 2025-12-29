# C4.utilities.py

from pathlib import Path
import shutil
from PPO.ppo_training_phases_config import TRAINING_PHASES

def reset_dir(path):
    p = Path(path).resolve()
    if str(p) in ("/", "C:\\", "C:/"):
        raise RuntimeError(f"Refusing to delete root path: {p}")
    shutil.rmtree(p, ignore_errors=True)   # delete dir and all contents
    p.mkdir(parents=True, exist_ok=True)   # recreate empty dir
    
def params_for_phase(phase_name: str, cfg) -> dict:
    phase = TRAINING_PHASES[phase_name]
    p_raw = phase.get("params", {})

    p = {
        "lr":                   p_raw.get("lr",        3e-4),
        "clip":                 p_raw.get("clip",      0.20),
        "entropy":              p_raw.get("entropy",   0.0),
        "epochs":               p_raw.get("epochs",    4),
        "batch_size":           p_raw.get("batch_size",      256),
        "steps_per_update":     p_raw.get("steps_per_update", 512),
        "vf_clip":              p_raw.get("vf_clip",   0.2),
        "max_grad_norm":        p_raw.get("max_grad_norm", 0.5),
        "target_kl":            p_raw.get("target_kl", 0.5),
        "temperature":          p_raw.get("temperature", 1.0),
        "vf_coef":              p_raw.get("vf_coef",   0.01),
        "distill_coef":         p_raw.get("distill_coef",   0.00),
        "mentor_depth":         p_raw.get("mentor_depth",   1),     
        "mentor_prob":          p_raw.get("mentor_prob",   0.1),   
        "mentor_coef":          p_raw.get("mentor_coef",   0.1),   
    }

    # --- pass through heuristic knobs untouched ---
    for k in ("center_start", "guard_prob", "win_now_prob",
              "guard_ply_min", "guard_ply_max"):
        if k in p_raw:
            p[k] = p_raw[k]

    return p

def make_hof_metascores(meta, ENSEMBLE, decimals=6, var_name="HOF_METASCORES"):
    """
    Build a pasteable Python dict literal from a meta dataframe/series and an ENSEMBLE list,
    sorted by score descending.

    Expects meta to contain scores indexed by model key like "MIX_9" (derived from filename stem),
    or alternatively to contain a column with those keys.
    """
    import os
    import pandas as pd

    # --- normalize meta -> a Series: index = model key (e.g. "MIX_9"), values = score ---
    if isinstance(meta, pd.Series):
        s = meta.copy()
    elif isinstance(meta, pd.DataFrame):
        # If it's a 1-col df, use that column; else try common names, else take the first numeric col.
        if meta.shape[1] == 1:
            s = meta.iloc[:, 0].copy()
        else:
            for col in ("meta", "score", "metascore"):
                if col in meta.columns:
                    s = meta[col].copy()
                    break
            else:
                num_cols = meta.select_dtypes(include="number").columns.tolist()
                if not num_cols:
                    raise ValueError("meta DataFrame has no numeric columns to use as scores.")
                s = meta[num_cols[0]].copy()

        # If index is not the model key, try to use a key/name column as index.
        if not isinstance(s.index, pd.Index) or s.index.dtype == "int64":
            for key_col in ("name", "model", "id", "key"):
                if key_col in meta.columns:
                    s.index = meta[key_col].astype(str)
                    break

    else:
        raise TypeError("meta must be a pandas Series or DataFrame.")

    s.index = s.index.astype(str)

    # --- map each ENSEMBLE path -> score via filename stem ("MIX_9.pt" -> "MIX_9") ---
    rows = []
    for path in ENSEMBLE:
        key = os.path.splitext(os.path.basename(path))[0]
        if key not in s.index:
            raise KeyError(f"Key '{key}' (from '{path}') not found in meta index.")
        rows.append((path, float(s.loc[key])))

    # --- sort by score desc, then by path for stability ---
    rows.sort(key=lambda x: (-x[1], x[0]))

    # --- build pasteable string ---
    lines = [f"{var_name} = {{"]
    for path, score in rows:
        lines.append(f'    "{path}": {score:.{decimals}f},')
    lines.append("}")
    return "\n".join(lines)

