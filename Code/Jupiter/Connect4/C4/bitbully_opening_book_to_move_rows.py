#!/usr/bin/env python3
"""
Export BitBully opening-book positions into the row format used by
`c4_move_data_generator.py`.

This version is batch-writing and memory-safe for large books:
- it does NOT keep the whole dataset in RAM
- it flushes rows to disk every N rows
- it splits Excel output into multiple files automatically

Output row format
-----------------
One row per *position before the move*, with columns compatible with your
existing pipeline:

- label
- reward
- game      (synthetic position id)
- ply       (1-based move number to be played from this position)
- player    (side to move: 1 or 2)
- action    (chosen best move)
- optional p0..p6 (uniform over tied-best legal moves)
- optional book_value
- optional s0..s6 raw BitBully move scores
- board cells 0-0 .. 5-6 (top row first, same as your generator)

Dependencies
------------
    pip install bitbully bitbully-databases pandas openpyxl numpy pyarrow

Typical Spyder usage
--------------------
Set RUN_HARDCODED = True and edit HARDCODED_CFG below, then just press Run.

Typical CLI usage
-----------------
    python bitbully_opening_book_to_move_rows_batched.py \
        --book 12-ply-dist \
        --output BitBully_12ply_rows.xlsx \
        --store-policy-probs \
        --store-book-value
"""

from __future__ import annotations

import argparse
import random
import re
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


ROWS = 6
COLS = 7
CENTER_COL = 3
CENTER_ORDER = (3, 4, 2, 5, 1, 6, 0)


# -----------------------------------------------------------------------------
# One-click Spyder run configuration
# -----------------------------------------------------------------------------
RUN_HARDCODED = True
HARDCODED_CFG = dict(
    book_name="12-ply-dist",
    output_path=Path("BitBully_12ply_rows.xlsx"),
    min_tokens=0,
    max_tokens=None,              # if None, inferred as book_nply - 1
    label="BB_12ply_dist",
    dedupe_mode="mirror",
    store_policy_probs=True,
    store_book_value=True,
    store_scores=False,
    action_mode="center",
    seed=666,
    merge_existing=False,
    progress_every=1000,
    flush_rows=100_000,           # rows kept in RAM before flush
    excel_chunk_rows=500_000,     # rows per Excel file
)


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------

def _infer_book_nply(book_name: str) -> Optional[int]:
    m = re.search(r"(\d+)-ply", str(book_name))
    return int(m.group(1)) if m else None


def _mirror_board(board: np.ndarray) -> np.ndarray:
    return board[:, ::-1]


def _board_key(board: np.ndarray, dedupe_mode: str = "mirror") -> Tuple[int, ...]:
    flat = tuple(int(x) for x in board.reshape(-1))
    if dedupe_mode in ("none", "exact"):
        return flat
    if dedupe_mode == "mirror":
        mir = tuple(int(x) for x in _mirror_board(board).reshape(-1))
        return min(flat, mir)
    raise ValueError(f"Unknown dedupe_mode={dedupe_mode!r}")


def _legal_cols(board: np.ndarray) -> List[int]:
    return [c for c in range(COLS) if int(board[0, c]) == 0]


def _lowest_empty_row(board: np.ndarray, col: int) -> int:
    for r in range(ROWS - 1, -1, -1):
        if int(board[r, col]) == 0:
            return r
    return -1


def _has_four_from(board: np.ndarray, row: int, col: int, token: int) -> bool:
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        cnt = 1

        rr, cc = row + dr, col + dc
        while 0 <= rr < ROWS and 0 <= cc < COLS and int(board[rr, cc]) == token:
            cnt += 1
            rr += dr
            cc += dc

        rr, cc = row - dr, col - dc
        while 0 <= rr < ROWS and 0 <= cc < COLS and int(board[rr, cc]) == token:
            cnt += 1
            rr -= dr
            cc -= dc

        if cnt >= 4:
            return True
    return False


def _apply_move(board: np.ndarray, col: int, player: int) -> Tuple[np.ndarray, int]:
    r = _lowest_empty_row(board, col)
    if r < 0:
        raise ValueError(f"Illegal move: column {col} is full")
    nxt = np.array(board, copy=True)
    nxt[r, col] = int(player)
    return nxt, r


def _board_to_cells(board: np.ndarray) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for r in range(ROWS):
        for c in range(COLS):
            out[f"{r}-{c}"] = int(board[r, c])
    return out


def _board_for_bitbully_db(board_top_first: np.ndarray) -> List[List[int]]:
    """
    Convert from your board layout (6x7, top row first) to the orientation used
    in the bitbully-databases examples (bottom row first).
    """
    return board_top_first[::-1, :].astype(int).tolist()


def _uniform_best_probs(scores: Sequence[int], legal_cols: Sequence[int]) -> np.ndarray:
    probs = np.zeros(COLS, dtype=np.float64)
    if not legal_cols:
        return probs

    best = max(int(scores[c]) for c in legal_cols)
    best_cols = [int(c) for c in legal_cols if int(scores[c]) == best]
    if not best_cols:
        return probs

    p = 1.0 / float(len(best_cols))
    for c in best_cols:
        probs[c] = p
    return probs


def _choose_action_from_probs(probs: np.ndarray, action_mode: str, rng: random.Random) -> int:
    best_cols = [c for c in range(COLS) if float(probs[c]) > 0.0]
    if not best_cols:
        return CENTER_COL

    mode = str(action_mode).lower()
    if mode == "sample":
        total = float(probs.sum())
        if total <= 0.0:
            return int(best_cols[0])
        x = rng.random() * total
        acc = 0.0
        for c in range(COLS):
            acc += float(probs[c])
            if x <= acc:
                return int(c)
        return int(best_cols[-1])

    if mode == "left":
        return int(min(best_cols))

    # default: center-first
    for c in CENTER_ORDER:
        if c in best_cols:
            return int(c)
    return int(best_cols[0])


def _records_to_dataframe(records: List[Dict[str, Any]]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()

    df = pd.DataFrame.from_records(records)
    board_cols = [f"{r}-{c}" for r in range(ROWS) for c in range(COLS)]
    head_cols = [c for c in ["label", "reward", "game", "ply"] if c in df.columns]
    extra_cols = [c for c in df.columns if c not in set(head_cols + board_cols)]
    return df[head_cols + sorted(extra_cols) + [c for c in board_cols if c in df.columns]]


# -----------------------------------------------------------------------------
# Batched file writing
# -----------------------------------------------------------------------------

def _list_matching_parts(output_path: Path) -> List[Path]:
    stem = output_path.stem
    suffix = output_path.suffix
    parent = output_path.parent
    return sorted(parent.glob(f"{stem}_part*.{suffix.lstrip('.')}"))


def _cleanup_existing_parts(output_path: Path) -> None:
    for p in _list_matching_parts(output_path):
        try:
            p.unlink()
        except FileNotFoundError:
            pass


def _write_table_batch(
    df: pd.DataFrame,
    output_path: Path,
    batch_idx: int,
    *,
    sheet_name: str = "Sheet1",
    excel_chunk_rows: int = 500_000,
) -> int:
    """
    Write one batch DataFrame.

    Returns the next batch index to use.

    For .xlsx this may split one batch into multiple files if needed.
    Files are named like:
        <stem>_part000001.xlsx
        <stem>_part000002.xlsx
        ...
    """
    output_path = Path(output_path)

    if df is None or len(df) == 0:
        return batch_idx

    suffix = output_path.suffix.lower()
    stem = output_path.stem
    parent = output_path.parent

    if suffix == ".parquet":
        part_path = parent / f"{stem}_part{batch_idx:06d}.parquet"
        df.to_parquet(part_path, index=False)
        print(f"Wrote {len(df):,} rows -> {part_path}")
        return batch_idx + 1

    if suffix == ".csv":
        part_path = parent / f"{stem}_part{batch_idx:06d}.csv"
        df.to_csv(part_path, index=False)
        print(f"Wrote {len(df):,} rows -> {part_path}")
        return batch_idx + 1

    if suffix == ".xlsx":
        n = len(df)
        for start in range(0, n, excel_chunk_rows):
            chunk = df.iloc[start:start + excel_chunk_rows].copy()
            part_path = parent / f"{stem}_part{batch_idx:06d}.xlsx"
            with pd.ExcelWriter(part_path, engine="openpyxl", mode="w") as w:
                chunk.to_excel(w, sheet_name=sheet_name, index=False)
            print(f"Wrote {len(chunk):,} rows -> {part_path}")
            batch_idx += 1
        return batch_idx

    raise ValueError(f"Unsupported output format: {output_path.suffix}")


# -----------------------------------------------------------------------------
# Main export
# -----------------------------------------------------------------------------

def export_bitbully_book_rows(
    *,
    book_name: str,
    output_path: Path,
    min_tokens: int = 0,
    max_tokens: Optional[int] = None,
    label: Optional[str] = None,
    dedupe_mode: str = "mirror",
    store_policy_probs: bool = True,
    store_book_value: bool = False,
    store_scores: bool = False,
    action_mode: str = "center",
    seed: int = 666,
    merge_existing: bool = False,
    progress_every: int = 1000,
    flush_rows: int = 100_000,
    excel_chunk_rows: int = 500_000,
) -> Dict[str, Any]:
    try:
        import bitbully_databases as bbd
        from bitbully import bitbully_core as bbc
    except Exception as e:
        raise RuntimeError(
            "This script requires 'bitbully' and 'bitbully-databases'. "
            "Install them with: pip install bitbully bitbully-databases"
        ) from e

    if merge_existing:
        raise ValueError(
            "merge_existing=True is not supported in batched mode. "
            "Use a fresh base output name or merge later."
        )

    inferred_nply = _infer_book_nply(book_name)
    if max_tokens is None:
        if inferred_nply is None:
            raise ValueError(
                "Could not infer book ply horizon from the book name. "
                "Please pass max_tokens explicitly."
            )
        max_tokens = inferred_nply - 1

    if max_tokens < min_tokens:
        raise ValueError("max_tokens must be >= min_tokens")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _cleanup_existing_parts(output_path)

    db_path = bbd.BitBullyDatabases.get_database_path(book_name)
    solver = bbc.BitBullyCore(db_path)
    db = bbd.BitBullyDatabases(book_name) if store_book_value else None

    rng = random.Random(seed)
    row_label = label or f"BB_{book_name}"

    empty = np.zeros((ROWS, COLS), dtype=np.int8)
    q: Deque[Tuple[np.ndarray, Tuple[int, ...]]] = deque()
    q.append((empty, tuple()))

    seen = {_board_key(empty, dedupe_mode=dedupe_mode)}
    records: List[Dict[str, Any]] = []

    n_processed = 0
    n_exported = 0
    total_written = 0
    batch_idx = 1

    while q:
        board, moves = q.popleft()
        tokens = len(moves)
        legal = _legal_cols(board)
        if not legal:
            continue

        player_to_move = 1 if (tokens % 2 == 0) else 2

        # Export current non-terminal position if it lies in the requested token window.
        if min_tokens <= tokens <= max_tokens:
            core_board = bbc.BoardCore()
            if moves:
                ok = core_board.setBoard(list(moves))
                if not ok:
                    raise RuntimeError(f"BitBully rejected move sequence: {moves}")

            scores = list(solver.scoreMoves(core_board))
            probs = _uniform_best_probs(scores, legal)
            action = _choose_action_from_probs(probs, action_mode=action_mode, rng=rng)

            row: Dict[str, Any] = {
                "label": str(row_label),
                "reward": 0.0,
                "game": int(n_exported),   # synthetic id, one row == one synthetic position id
                "ply": int(tokens + 1),
                "player": int(player_to_move),
                "action": int(action),
                **_board_to_cells(board),
            }

            if store_policy_probs:
                for a in range(COLS):
                    row[f"p{a}"] = float(probs[a])

            if store_book_value and db is not None:
                row["book_value"] = db.get_book_value(_board_for_bitbully_db(board))

            if store_scores:
                for a in range(COLS):
                    row[f"s{a}"] = int(scores[a])

            records.append(row)
            n_exported += 1

            if progress_every > 0 and (n_exported % progress_every == 0):
                print(
                    f"exported {n_exported:,} rows "
                    f"(processed {n_processed:,} positions, queue={len(q):,}, in_ram={len(records):,})"
                )

            # Flush to disk before RAM grows too large.
            if len(records) >= flush_rows:
                df_chunk = _records_to_dataframe(records)
                batch_idx = _write_table_batch(
                    df_chunk,
                    output_path=output_path,
                    batch_idx=batch_idx,
                    excel_chunk_rows=excel_chunk_rows,
                )
                total_written += len(df_chunk)
                records.clear()

        # Expand only until max_tokens so exported move remains within horizon.
        if tokens >= max_tokens:
            n_processed += 1
            continue

        for c in CENTER_ORDER:
            if c not in legal:
                continue
            child, row_idx = _apply_move(board, c, player_to_move)

            # Only the side who just moved could have created a terminal board.
            if _has_four_from(child, row_idx, c, player_to_move):
                continue
            if not _legal_cols(child):
                continue

            key = _board_key(child, dedupe_mode=dedupe_mode)
            if key in seen:
                continue
            seen.add(key)
            q.append((child, moves + (int(c),)))

        n_processed += 1

    # Final flush.
    if records:
        df_chunk = _records_to_dataframe(records)
        batch_idx = _write_table_batch(
            df_chunk,
            output_path=output_path,
            batch_idx=batch_idx,
            excel_chunk_rows=excel_chunk_rows,
        )
        total_written += len(df_chunk)
        records.clear()

    summary = {
        "book_name": book_name,
        "output_base": str(output_path),
        "rows_written": int(total_written),
        "files_written": int(batch_idx - 1),
        "positions_processed": int(n_processed),
        "positions_exported": int(n_exported),
        "dedupe_mode": dedupe_mode,
        "flush_rows": int(flush_rows),
        "excel_chunk_rows": int(excel_chunk_rows),
    }
    return summary


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export BitBully opening-book positions into C4 move-row format")
    p.add_argument("--book", default="12-ply-dist", help="BitBully database name, e.g. 8-ply or 12-ply-dist")
    p.add_argument("--output", required=True, help="Base output file (.xlsx, .csv, or .parquet)")
    p.add_argument("--min-tokens", type=int, default=0, help="Minimum number of tokens in exported positions")
    p.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Maximum number of tokens in exported positions. Default: book_nply - 1",
    )
    p.add_argument("--label", default=None, help="Row label. Default: BB_<book_name>")
    p.add_argument(
        "--dedupe-mode",
        default="mirror",
        choices=["mirror", "exact", "none"],
        help="How to dedupe traversed positions",
    )
    p.add_argument("--seed", type=int, default=666, help="RNG seed used when action-mode=sample")
    p.add_argument(
        "--action-mode",
        default="center",
        choices=["center", "left", "sample"],
        help="How to choose one action among tied-best legal moves",
    )
    p.add_argument("--store-policy-probs", action="store_true", help="Store p0..p6 uniform over tied-best legal moves")
    p.add_argument("--store-book-value", action="store_true", help="Store current-position book value")
    p.add_argument("--store-scores", action="store_true", help="Store raw move scores as s0..s6")
    p.add_argument("--progress-every", type=int, default=1000, help="Print progress every N exported rows (0 disables)")
    p.add_argument("--flush-rows", type=int, default=100_000, help="Rows kept in RAM before flush")
    p.add_argument("--excel-chunk-rows", type=int, default=500_000, help="Rows per Excel file")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    summary = export_bitbully_book_rows(
        book_name=args.book,
        output_path=Path(args.output),
        min_tokens=int(args.min_tokens),
        max_tokens=args.max_tokens,
        label=args.label,
        dedupe_mode=args.dedupe_mode,
        store_policy_probs=bool(args.store_policy_probs),
        store_book_value=bool(args.store_book_value),
        store_scores=bool(args.store_scores),
        action_mode=args.action_mode,
        seed=int(args.seed),
        merge_existing=False,
        progress_every=int(args.progress_every),
        flush_rows=int(args.flush_rows),
        excel_chunk_rows=int(args.excel_chunk_rows),
    )

    print("Done.")
    for k, v in summary.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    if RUN_HARDCODED:
        summary = export_bitbully_book_rows(**HARDCODED_CFG)
        print("Done.")
        for k, v in summary.items():
            print(f"{k}: {v}")
    else:
        main()
