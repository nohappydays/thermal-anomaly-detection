"""Parse TCView exports into the common DataFrame schema.

Supported inputs:
  * .xlsx  — saved from TCView (typical Excel team output)
  * .csv   — already-converted exports
  * .numbers — Apple Numbers format (requires `numbers-parser`, optional)

The TCView export format is:
  Dot    : time | Temperature                            @ 1 Hz
  Plane  : time | Lowest Temperature | Highest Temperature @ 1 Hz
  Line   : time | <per-pixel columns>                    @ 1 Hz   (not seen yet)

Usage:
    from excel_loader import load_tcview
    df = load_tcview("12May_CPUonly.xlsx", condition="C1", session_id="S001")
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd

log = logging.getLogger("excel_loader")

_NUMERIC_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _strip_unit(value) -> float:
    """Convert '23.4°C' / 23.4 / None → float (NaN on failure)."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return float("nan")
    if isinstance(value, (int, float)):
        return float(value)
    m = _NUMERIC_RE.search(str(value))
    return float(m.group()) if m else float("nan")


def _parse_time(value) -> Optional[datetime]:
    if isinstance(value, datetime):
        return value
    if value is None:
        return None
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        try:
            return datetime.strptime(str(value), "%Y-%m-%d %H:%M:%S")
        except ValueError:
            return None


@dataclass
class TCViewExport:
    """Validated, normalized TCView measurement export."""

    df: pd.DataFrame
    mode: Literal["dot", "plane", "line", "unknown"]
    sample_rate_hz: float
    duration_s: float
    n_samples: int
    source: Path
    condition: str
    session_id: str
    laptop_id: str

    def summary(self) -> str:
        return (
            f"{self.source.name}: mode={self.mode} "
            f"samples={self.n_samples} duration={self.duration_s/60:.1f}min "
            f"rate={self.sample_rate_hz:.2f}Hz condition={self.condition} "
            f"laptop={self.laptop_id}"
        )


def _load_raw_table(path: Path) -> pd.DataFrame:
    """Load any supported format → raw DataFrame, no parsing yet."""
    suffix = path.suffix.lower()
    if suffix == ".xlsx":
        return pd.read_excel(path, engine="openpyxl")
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".numbers":
        try:
            from numbers_parser import Document  # type: ignore
        except ImportError as e:
            raise ImportError(
                "Reading .numbers files requires `pip install numbers-parser`."
            ) from e
        doc = Document(str(path))
        table = doc.sheets[0].tables[0]
        rows = table.rows(values_only=True)
        return pd.DataFrame(rows[1:], columns=rows[0])
    raise ValueError(f"Unsupported file type: {path.suffix}")


def _detect_mode(columns: list[str]) -> Literal["dot", "plane", "line", "unknown"]:
    cols_lc = [str(c).strip().lower() for c in columns]
    has_low = any("low" in c for c in cols_lc)
    has_high = any("high" in c for c in cols_lc)
    if has_low and has_high:
        return "plane"
    has_temp = any(c == "temperature" or c.endswith("temperature") for c in cols_lc)
    if has_temp and len(columns) == 2:
        return "dot"
    if len(columns) > 4:
        return "line"
    return "unknown"


def load_tcview(
    path: str | Path,
    condition: str = "UNLABELED",
    session_id: Optional[str] = None,
    laptop_id: str = "unknown",
) -> TCViewExport:
    """Load a TCView measurement export and return a validated TCViewExport.

    The returned DataFrame is normalized to these columns:
      timestamp (datetime64[ns]), t_s (float, seconds from session start),
      T_max (float), T_min (float, NaN for Dot mode), T_range (float, NaN for Dot)
    plus passthrough columns:
      condition, session_id, laptop_id
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    raw = _load_raw_table(path)
    if raw.empty:
        raise ValueError(f"{path}: empty table")

    mode = _detect_mode(list(raw.columns))
    log.info("Detected mode=%s for %s (%d rows, columns=%s)",
             mode, path.name, len(raw), list(raw.columns))

    raw.columns = [str(c).strip() for c in raw.columns]
    time_col = next((c for c in raw.columns if "time" in c.lower()), raw.columns[0])

    timestamps = raw[time_col].map(_parse_time)
    if timestamps.isna().all():
        raise ValueError(f"{path}: could not parse any timestamps in column {time_col!r}")

    if mode == "plane":
        low_col = next(c for c in raw.columns if "low" in c.lower())
        high_col = next(c for c in raw.columns if "high" in c.lower())
        t_min = raw[low_col].map(_strip_unit).astype(float)
        t_max = raw[high_col].map(_strip_unit).astype(float)
    elif mode == "dot":
        temp_col = next(c for c in raw.columns if "temperature" in c.lower())
        t_max = raw[temp_col].map(_strip_unit).astype(float)
        t_min = pd.Series([np.nan] * len(raw))
    else:
        log.warning("%s: unknown mode — treating column %r as T_max",
                    path.name, raw.columns[1])
        t_max = raw[raw.columns[1]].map(_strip_unit).astype(float)
        t_min = pd.Series([np.nan] * len(raw))

    t0 = timestamps.dropna().iloc[0]
    t_s = timestamps.map(lambda x: (x - t0).total_seconds() if x is not None else np.nan)

    df = pd.DataFrame({
        "timestamp": timestamps,
        "t_s": t_s,
        "T_max": t_max,
        "T_min": t_min,
        "T_range": t_max - t_min,
        "condition": condition,
        "session_id": session_id or path.stem,
        "laptop_id": laptop_id,
    })

    n_before = len(df)
    df = df.dropna(subset=["timestamp", "t_s", "T_max"]).reset_index(drop=True)
    if len(df) < n_before:
        log.warning("%s: dropped %d unparseable rows", path.name, n_before - len(df))

    if df.empty:
        raise ValueError(f"{path}: no valid rows after parsing")

    duration_s = float(df["t_s"].iloc[-1] - df["t_s"].iloc[0])
    rate_hz = (len(df) - 1) / duration_s if duration_s > 0 else 0.0

    return TCViewExport(
        df=df,
        mode=mode,
        sample_rate_hz=rate_hz,
        duration_s=duration_s,
        n_samples=len(df),
        source=path,
        condition=condition,
        session_id=session_id or path.stem,
        laptop_id=laptop_id,
    )


def load_many(
    sources: list[dict],
) -> list[TCViewExport]:
    """Load multiple TCView exports.

    `sources` is a list of dicts: {path, condition, session_id, laptop_id}.
    """
    out: list[TCViewExport] = []
    for s in sources:
        out.append(load_tcview(
            s["path"],
            condition=s.get("condition", "UNLABELED"),
            session_id=s.get("session_id"),
            laptop_id=s.get("laptop_id", "unknown"),
        ))
    return out


def concat(exports: list[TCViewExport]) -> pd.DataFrame:
    """Stack multiple exports' DataFrames into one tidy long table."""
    return pd.concat([e.df for e in exports], ignore_index=True)


if __name__ == "__main__":
    import argparse
    import sys

    p = argparse.ArgumentParser(description="Load and summarize TCView Excel/CSV exports")
    p.add_argument("paths", nargs="+", type=Path)
    p.add_argument("--condition", default="UNLABELED")
    p.add_argument("--laptop", default="unknown")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    for path in args.paths:
        try:
            exp = load_tcview(path, condition=args.condition, laptop_id=args.laptop)
            print(exp.summary())
            print(exp.df.head(3).to_string(index=False))
            print(f"  T_max: mean={exp.df['T_max'].mean():.2f}  max={exp.df['T_max'].max():.2f}")
            if exp.mode == "plane":
                print(f"  T_min: mean={exp.df['T_min'].mean():.2f}  min={exp.df['T_min'].min():.2f}")
            print()
        except Exception as e:
            print(f"[ERROR] {path}: {e}", file=sys.stderr)
            sys.exit(1)
