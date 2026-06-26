"""
L4 — Deterministic dimension scorer.

Contract: same signals + same rubric → same output.  No LLM involved.

Public API
----------
score(signals, rubric)  →  dict  (dimension labels + confidence_score)
get_rubric()            →  dict  (cached rubric_v1.yaml)

SignalUnit fields
-----------------
trait       : L0 dimension name
value       : must be in rubric allowed_values for that dimension
weight      : relative evidence weight (default 1.0)
source_ref  : human-readable provenance string (e.g. "missed_deadline:opt_c")
"""
from __future__ import annotations

import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SCORER_VERSION = "v1"

_RUBRIC_PATH = Path(__file__).parent.parent / "traits" / "rubric_v1.yaml"


@dataclass
class SignalUnit:
    trait: str
    value: str
    weight: float = 1.0
    source_ref: str = ""


# ---------------------------------------------------------------------------
# Rubric loader (module-level cache; safe for read-only use)
# ---------------------------------------------------------------------------

_rubric_cache: dict | None = None


def get_rubric() -> dict:
    global _rubric_cache
    if _rubric_cache is None:
        with open(_RUBRIC_PATH) as f:
            _rubric_cache = yaml.safe_load(f)
    return _rubric_cache


# ---------------------------------------------------------------------------
# Core scoring
# ---------------------------------------------------------------------------

def score(signals: list[SignalUnit], rubric: dict) -> dict[str, Any]:
    """
    Aggregate SignalUnits into final dimension labels.

    Returns a flat dict:
      {dim_name: label, ..., "_unknown_dims": [dim_names_with_no_signals],
       "confidence_score": float}

    "confidence_score" is SCORER MEASUREMENT CONFIDENCE (mean signal
    coverage fraction) — NOT a student personality trait.
    """
    dims = rubric["dimensions"]

    # Group signals by trait
    by_trait: dict[str, list[SignalUnit]] = {}
    for s in signals:
        by_trait.setdefault(s.trait, []).append(s)

    result: dict[str, Any] = {}
    unknown_dims: list[str] = []
    coverage_fractions: list[float] = []

    for dim_name, dim_def in dims.items():
        if dim_def.get("type") == "derived":
            continue

        trait_signals = by_trait.get(dim_name, [])

        if not trait_signals:
            unknown_dims.append(dim_name)
            coverage_fractions.append(0.0)
            # default_value is for DISPLAY only — do not assign here;
            # callers must check _unknown_dims before using the value.
            result[dim_name] = dim_def.get("default_value", "")
            continue

        coverage_fractions.append(min(1.0, len(trait_signals) / 3))

        dim_type = dim_def.get("type", "categorical")

        if dim_type == "categorical":
            result[dim_name] = _plurality(trait_signals)
        elif dim_type == "ordinal":
            result[dim_name] = _ordinal_mean(trait_signals, dim_def)
        else:
            result[dim_name] = dim_def.get("default_value", "")

    result["_unknown_dims"] = unknown_dims
    result["confidence_score"] = (
        round(sum(coverage_fractions) / len(coverage_fractions), 3)
        if coverage_fractions else 0.0
    )

    return result


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def _plurality(signals: list[SignalUnit]) -> str:
    """Return the value with the highest total weight; ties → first-seen."""
    totals: dict[str, float] = {}
    for s in signals:
        totals[s.value] = totals.get(s.value, 0.0) + s.weight
    return max(totals, key=totals.__getitem__)


def _ordinal_mean(signals: list[SignalUnit], dim_def: dict) -> str:
    """Weighted mean of ordinal positions → nearest allowed value."""
    allowed: list[str] = dim_def["allowed_values"]
    level_map = {v: i for i, v in enumerate(allowed)}
    max_level = len(allowed) - 1

    total_weight = sum(s.weight for s in signals)
    if total_weight == 0:
        return allowed[max_level // 2]

    weighted_sum = sum(
        level_map.get(s.value, max_level // 2) * s.weight
        for s in signals
    )
    mean_pos = weighted_sum / total_weight
    idx = round(mean_pos)
    idx = max(0, min(max_level, idx))
    return allowed[idx]
