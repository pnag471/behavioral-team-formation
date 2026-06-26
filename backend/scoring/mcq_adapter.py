"""
MCQ adapter — converts bank_v1.yaml option choices to SignalUnits.

Public API
----------
get_bank()                  →  dict  (cached bank_v1.yaml)
responses_to_signals(responses)  →  list[SignalUnit]
get_scenarios_for_client()  →  list[dict]  (anonymous option IDs; no trait leakage)
"""
from __future__ import annotations

import yaml
from pathlib import Path

from scoring.dimension_scorer import SignalUnit

_BANK_PATH = Path(__file__).parent.parent / "questions" / "bank_v1.yaml"

_bank_cache: dict | None = None


def get_bank() -> dict:
    global _bank_cache
    if _bank_cache is None:
        with open(_BANK_PATH) as f:
            _bank_cache = yaml.safe_load(f)
    return _bank_cache


def _build_option_map(bank: dict) -> dict[str, dict[str, list[dict]]]:
    """scenario_id → {anonymous_id → [signal_defs]}"""
    result: dict[str, dict[str, list[dict]]] = {}
    for scenario in bank["scenarios"]:
        opt_map: dict[str, list[dict]] = {}
        for opt in scenario["options"]:
            opt_map[opt["anonymous_id"]] = opt.get("signals", [])
        result[scenario["id"]] = opt_map
    return result


def responses_to_signals(responses: list) -> list[SignalUnit]:
    """
    Convert a list of AssessmentResponse-like objects to SignalUnits.

    Each response must have .scenario_id and .option_id (anonymous, e.g. 'opt_c').
    Unknown scenario_id or option_id → silently skipped (no signal emitted).
    """
    bank = get_bank()
    option_map = _build_option_map(bank)

    signals: list[SignalUnit] = []
    for resp in responses:
        per_scenario = option_map.get(resp.scenario_id, {})
        sig_defs = per_scenario.get(resp.option_id, [])
        for sig_def in sig_defs:
            signals.append(SignalUnit(
                trait=sig_def["trait"],
                value=sig_def["value"],
                weight=1.0,
                source_ref=f"{resp.scenario_id}:{resp.option_id}",
            ))
    return signals


def get_scenarios_for_client() -> list[dict]:
    """
    Return scenarios with anonymous option IDs (opt_a … opt_d) so
    the trait-value mapping is not exposed to the client.
    """
    bank = get_bank()
    result: list[dict] = []
    for scenario in bank["scenarios"]:
        options = [
            {"id": opt["anonymous_id"], "label": opt["text"]}
            for opt in scenario["options"]
        ]
        result.append({
            "id": scenario["id"],
            "title": scenario["title"],
            "description": scenario["description"],
            "options": options,
        })
    return result
