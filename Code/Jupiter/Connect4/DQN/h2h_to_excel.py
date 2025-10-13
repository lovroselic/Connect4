# DQN.h2h_to_excel.py

import os
import pandas as pd

EXCEL_PATH = "H2h.xlsx"
SHEET_NAME = "H2H"
COLS = ["Games", "A_path", "B_path", "A_wins", "A_loses", "draws", "A-score", "95% CI"]

def _result_to_row(res: dict) -> dict:
    ci = res.get("A_score_CI95")
    ci_str = ""
    if ci is not None and len(ci) == 2:
        lo, hi = ci
        ci_str = f"{lo:.3f}–{hi:.3f}"

    return {
        "Games": int(res.get("games", 0)),
        "A_path": res.get("A_path", ""),
        "B_path": res.get("B_path", ""),
        "A_wins": int(res.get("A_wins", 0)),
        "A_loses": int(res.get("A_losses", res.get("A_loses", 0))),
        "draws": int(res.get("draws", 0)),
        "A-score": float(res.get("A_score_rate", 0.0)),
        "95% CI": ci_str,
    }

def append_h2h_to_excel(res: dict, excel_path: str = EXCEL_PATH, sheet_name: str = SHEET_NAME):
    row = _result_to_row(res)

    if os.path.exists(excel_path):
        try:
            df = pd.read_excel(excel_path, sheet_name=sheet_name)
        except ValueError:
            # file exists but sheet doesn't
            df = pd.DataFrame(columns=COLS)
    else:
        df = pd.DataFrame(columns=COLS)

    df = df[COLS]
    df_new = pd.concat([df, pd.DataFrame([row], columns=COLS)], ignore_index=True)

    with pd.ExcelWriter(excel_path, engine="openpyxl", mode="w") as writer:
        df_new.to_excel(writer, index=False, sheet_name=sheet_name)
