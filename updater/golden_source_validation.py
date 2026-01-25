from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, Sequence

import pandas as pd


@dataclass(frozen=True)
class GoldenSourceCheck:
    label: str
    path: str
    key_columns: Sequence[str]
    compare_columns: Sequence[str]


def _normalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out.columns = [str(c) for c in out.columns]
    for col in out.columns:
        if "date" in col.lower():
            out[col] = pd.to_datetime(out[col], errors="coerce").dt.normalize()
    return out


def _compare_frames(current: pd.DataFrame, golden: pd.DataFrame, check: GoldenSourceCheck) -> None:
    missing_keys = [c for c in check.key_columns if c not in current.columns or c not in golden.columns]
    if missing_keys:
        raise ValueError(f"{check.label}: missing key columns {missing_keys}")

    missing_compare = [c for c in check.compare_columns if c not in current.columns or c not in golden.columns]
    if missing_compare:
        raise ValueError(f"{check.label}: missing compare columns {missing_compare}")
    compare_cols = list(check.compare_columns)

    current_view = current[list(dict.fromkeys(list(check.key_columns) + compare_cols))].copy()
    golden_view = golden[list(dict.fromkeys(list(check.key_columns) + compare_cols))].copy()

    current_view = current_view.sort_values(list(check.key_columns)).reset_index(drop=True)
    golden_view = golden_view.sort_values(list(check.key_columns)).reset_index(drop=True)

    if len(current_view) != len(golden_view):
        raise ValueError(
            f"{check.label}: row count mismatch (current={len(current_view)}, golden={len(golden_view)})"
        )

    if not current_view.equals(golden_view):
        diff = current_view.merge(
            golden_view,
            on=list(check.key_columns),
            how="outer",
            indicator=True,
            suffixes=("_current", "_golden"),
        )
        mismatched = diff[diff["_merge"] != "both"].head(10)
        raise ValueError(f"{check.label}: data mismatch. Sample diff:\n{mismatched}")


def validate_golden_source(
    checks: Iterable[GoldenSourceCheck],
    current_frames: dict[str, pd.DataFrame],
) -> list[str]:
    messages: list[str] = []

    for check in checks:
        current = current_frames.get(check.label)
        if current is None:
            messages.append(f"{check.label}: current data not provided; skipping.")
            continue

        if not os.path.exists(check.path):
            messages.append(f"{check.label}: golden source file not found at {check.path}; skipping.")
            continue

        golden = _normalize_frame(pd.read_parquet(check.path))
        current_norm = _normalize_frame(current)
        _compare_frames(current_norm, golden, check)
        messages.append(f"{check.label}: ✅ matches golden source.")

    return messages
