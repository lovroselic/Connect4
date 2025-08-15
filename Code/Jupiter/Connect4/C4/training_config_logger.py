# -*- coding: utf-8 -*-
"""
Created on Wed Aug 13 11:21:57 2025

@author: Uporabnik
"""

# C4.training_config_logger.py
# usage:: from C4.training_config_logger import export_training_config

from __future__ import annotations

import json
import os
from typing import Any, Dict, List

def _stringify(v: Any) -> str:
    """Readable, safe string for Excel/CSV cells."""
    try:
        import numpy as np  # optional
        if isinstance(v, (np.integer,)):
            return str(int(v))
        if isinstance(v, (np.floating,)):
            # avoid scientific notation noise
            return f"{float(v):.12g}"
        if isinstance(v, (np.ndarray,)):
            return json.dumps(v.tolist())
    except Exception:
        pass

    if isinstance(v, (list, tuple)):
        return json.dumps(list(v))
    if isinstance(v, (dict,)):
        # compact but readable JSON
        return json.dumps(v, separators=(",", ":"), ensure_ascii=False)
    return str(v)

def _flatten_config(config: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Flattens a config dict into rows: Section | Phase | Key | Value
    - If Section value is dict-like (e.g., TRAINING_PHASES), iterate phases
    - Otherwise write a single row for the scalar Section
    """
    rows: List[Dict[str, str]] = []

    for section, values in config.items():
        # TRAINING_PHASES-style nested dict
        if isinstance(values, dict):
            for phase_name, params in values.items():
                # params may be a dict (typical) or a scalar
                if isinstance(params, dict):
                    for k, v in params.items():
                        rows.append({
                            "Section": str(section),
                            "Phase": str(phase_name),
                            "Key": str(k),
                            "Value": _stringify(v),
                        })
                else:
                    rows.append({
                        "Section": str(section),
                        "Phase": str(phase_name),
                        "Key": "",
                        "Value": _stringify(params),
                    })
        else:
            rows.append({
                "Section": str(section),
                "Phase": "",
                "Key": "",
                "Value": _stringify(values),
            })

    return rows

def export_training_config(
    *,
    training_phases: Dict[str, Dict[str, Any]],
    lookahead_depth: int,
    num_episodes: int,
    batch_size: int,
    target_update_interval: int,
    log_dir: str,
    session_name: str,
    write_excel: bool = True,
    write_json: bool = True,
) -> Dict[str, str]:
    """
    Build a normalized table of the training configuration and save to disk.

    Returns a dict with paths to the written files (excel/csv/json where applicable).
    """
    os.makedirs(log_dir, exist_ok=True)

    # Assemble a single config mapping
    config_log: Dict[str, Any] = {
        "TRAINING_PHASES": training_phases,
        "lookahead_depth": lookahead_depth,
        "num_episodes": num_episodes,
        "batch_size": batch_size,
        "target_update_interval": target_update_interval,
    }

    # Optional JSON dump (faithful structure)
    out_paths: Dict[str, str] = {}
    if write_json:
        json_path = os.path.join(log_dir, f"DQN-{session_name}_training_config.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(config_log, f, indent=2, ensure_ascii=False)
        out_paths["json"] = json_path

    # Flatten for tabular export
    rows = _flatten_config(config_log)

    # Lazy import pandas to keep module light when not used
    try:
        import pandas as pd
    except Exception as e:
        print(e)
        # pandas missing: write CSV rows manually
        csv_path = os.path.join(log_dir, f"DQN-{session_name}_training_config.csv")
        import csv
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["Section", "Phase", "Key", "Value"])
            writer.writeheader()
            writer.writerows(rows)
        out_paths["csv"] = csv_path
        return out_paths

    df = pd.DataFrame(rows, columns=["Section", "Phase", "Key", "Value"])

    # Prefer Excel; fall back to CSV if engine missing
    if write_excel:
        excel_path = os.path.join(log_dir, f"DQN-{session_name}_training_config.xlsx")
        try:
            with pd.ExcelWriter(excel_path, engine="openpyxl", mode="w") as writer:
                df.to_excel(writer, sheet_name="config", index=False)
            out_paths["excel"] = excel_path
            return out_paths
        except Exception:
            # openpyxl not available or file in use → fallback to CSV
            pass

    csv_path = os.path.join(log_dir, f"DQN-{session_name}_training_config.csv")
    df.to_csv(csv_path, index=False)
    out_paths["csv"] = csv_path
    return out_paths
