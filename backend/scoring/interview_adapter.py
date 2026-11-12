"""
Interview adapter: converts LLM-extracted interview evidence to SignalUnits.

The interview counterpart of mcq_adapter.responses_to_signals. The LLM only
extracts evidence (a verbatim quote plus the dimension, indicator, polarity,
strength and confidence it assigns). This module validates that evidence
against the rubric and the transcript, and turns what survives into
SignalUnits for the deterministic scorer, dimension_scorer.score().

Pure module: no DB, no LLM, no network. Same inputs, same outputs.

Public API
----------
EvidenceItem                 one evidence item as extracted by the LLM
EvidenceDecision             what the adapter did with one item, and why
evidence_to_signals(items, transcript_student_text, rubric)
                             ->  (list[SignalUnit], list[EvidenceDecision])
normalize_text(text)         ->  str  (the normalization used by the verbatim check)
floor_dimensions(rubric)     ->  list[str]

Decision reasons
----------------
  rejected:unknown_dimension      dimension is not a non-derived rubric dimension
  rejected:indicator_not_allowed  indicator is not in that dimension's allowed_values
  rejected:unknown_strength       strength is not a key of rubric["evidence_strength"]
  rejected:unknown_polarity       polarity is not "supporting" or "contradicting"
  rejected:not_verbatim           quote not found in the student's turns
  excluded:contradicting          valid, but contradicting: never becomes a signal
  excluded:duplicate              quote is a substring of another kept quote
                                  for the same (dimension, indicator)
  excluded:over_cap               more than MAX_ITEMS_PER_INDICATOR for the same
                                  (dimension, indicator)
  accepted                        became exactly one SignalUnit

Confidence never affects a signal. item.confidence is read in exactly one
place, the optional review flag in _needs_review, and only when a threshold is
configured (off by default).

See docs/interview-path-plan.md, Part B2, and the TODO defaults in Part C.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from scoring.dimension_scorer import SignalUnit

# Decision doc §6: at most 2 counted evidence items per (dimension, indicator).
MAX_ITEMS_PER_INDICATOR = 2

POLARITIES = ("supporting", "contradicting")

# TODO(team-decision): per-evidence confidence threshold for needs_review
# (plan TODO-3). Default OFF (None): confidence is stored for reviewers to sort
# by, but never sets needs_review on its own, so no threshold number is
# invented. LLM self-reported confidence is uncalibrated. If the team picks a
# number, set it here or pass confidence_review_threshold= per call. Either
# way it only affects needs_review, never a signal or a weight.
CONFIDENCE_REVIEW_THRESHOLD: Optional[float] = None

STATUS_ACCEPTED = "accepted"
STATUS_REJECTED = "rejected"
STATUS_EXCLUDED = "excluded"


@dataclass(frozen=True)
class EvidenceItem:
    quote: str
    dimension: str
    indicator: str
    polarity: str
    strength: str
    confidence: Optional[float] = None


@dataclass(frozen=True)
class EvidenceDecision:
    item: EvidenceItem
    status: str                    # accepted | rejected | excluded
    reason: Optional[str]          # e.g. "not_verbatim"; None when accepted
    needs_review: bool
    quote_start: Optional[int]     # offset in the NORMALIZED student text; None if not found
    source_ref: Optional[str] = None   # set only when accepted; matches the SignalUnit

    @property
    def decision(self) -> str:
        """Full decision string, e.g. 'rejected:not_verbatim' or 'accepted'."""
        return self.status if self.reason is None else f"{self.status}:{self.reason}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_QUOTE_CHARS = str.maketrans({
    "‘": "'", "’": "'", "‚": "'", "‛": "'", "′": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"', "″": '"',
})
_WHITESPACE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """
    TODO-4 default: fold case, map curly quotes to straight ones, collapse
    every whitespace run to one space, strip the ends. Nothing else (no
    punctuation stripping, no fuzzy matching).
    """
    return _WHITESPACE.sub(" ", text.translate(_QUOTE_CHARS).casefold()).strip()


def floor_dimensions(rubric: dict) -> list[str]:
    """Floor dims, picked the same way the matcher does: by composition_rule AND type, not weight."""
    return sorted(
        name for name, d in rubric["dimensions"].items()
        if d.get("composition_rule") == "high_mean_with_floor" and d.get("type") == "ordinal"
    )


def _scorable_dimensions(rubric: dict) -> dict[str, dict]:
    return {
        name: d for name, d in rubric["dimensions"].items()
        if d.get("type") != "derived"
    }


def _validate(item: EvidenceItem, student_norm: str, rubric: dict
              ) -> tuple[Optional[str], Optional[int]]:
    """Return (reject_reason or None, quote_start or None). First failing check wins."""
    dims = _scorable_dimensions(rubric)
    strengths = rubric.get("evidence_strength") or {}

    if not isinstance(item.dimension, str) or item.dimension not in dims:
        return "unknown_dimension", None
    if not isinstance(item.indicator, str) or item.indicator not in dims[item.dimension]["allowed_values"]:
        return "indicator_not_allowed", None
    if not isinstance(item.strength, str) or item.strength not in strengths:
        return "unknown_strength", None
    # Not in the task's list of reasons, but an unrecognised polarity must not
    # silently count as "supporting". The response_schema enum should make
    # this unreachable in practice.
    if item.polarity not in POLARITIES:
        return "unknown_polarity", None

    quote_norm = normalize_text(item.quote) if isinstance(item.quote, str) else ""
    if not quote_norm:  # "" is a substring of everything
        return "not_verbatim", None
    start = student_norm.find(quote_norm)
    if start < 0:
        return "not_verbatim", None
    return None, start


def _needs_review(status: str, reason: Optional[str], item: EvidenceItem,
                  floors: set[str], threshold: Optional[float]) -> bool:
    if status == STATUS_REJECTED:
        return True
    if reason == "contradicting" and item.dimension in floors:
        return True
    if threshold is not None and isinstance(item.confidence, (int, float)):
        return item.confidence < threshold
    return False


def _source_ref(item: EvidenceItem, quote_start: int) -> str:
    # TODO(team-decision): source_ref format. This function has no DB ids, so
    # the ref is synthetic and deterministic: "dimension:indicator:quote_start"
    # (quote_start is the offset in the normalized student text). It is unique
    # among accepted items, because two kept items with the same dimension,
    # indicator and start would have one quote as a prefix of the other, and
    # dedupe drops one. The caller maps it to the real evidence_unit id
    # ("{run_id}:{evidence_unit_id}" in plan B2) once rows are written.
    return f"{item.dimension}:{item.indicator}:{quote_start}"


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def evidence_to_signals(
    items: list[EvidenceItem],
    transcript_student_text: str,
    rubric: dict,
    *,
    confidence_review_threshold: Optional[float] = CONFIDENCE_REVIEW_THRESHOLD,
) -> tuple[list[SignalUnit], list[EvidenceDecision]]:
    """
    Convert LLM evidence items into SignalUnits for score(), plus one decision per item.

    transcript_student_text must contain ONLY the student's turns (no
    interviewer text), so a quote of the interviewer is rejected as not_verbatim.

    Steps, in order (plan B2):
      1. validate; reject with a reason
      2. exclude contradicting items (never signals)
      3. drop substring duplicates, then cap at MAX_ITEMS_PER_INDICATOR per
         (dimension, indicator)
      4. map survivors to SignalUnit; weight comes ONLY from
         rubric["evidence_strength"][strength]["weight"]
      5. emit signals sorted by (dimension, quote_start, indicator, quote)

    Both returned lists are in a deterministic order that does not depend on
    the order of `items`. Decisions are sorted by (dimension, quote_start,
    indicator, quote, polarity, strength); rejected items without a
    quote_start sort after those with one.
    """
    student_norm = normalize_text(transcript_student_text or "")
    floors = set(floor_dimensions(rubric))
    strengths = rubric.get("evidence_strength") or {}

    # (item, status, reason, quote_start)
    records: list[tuple[EvidenceItem, str, Optional[str], Optional[int]]] = []
    candidates: list[tuple[EvidenceItem, int]] = []

    # 1 + 2: validate, then set contradicting items aside.
    for item in items:
        reason, start = _validate(item, student_norm, rubric)
        if reason is not None:
            records.append((item, STATUS_REJECTED, reason, start))
        elif item.polarity == "contradicting":
            records.append((item, STATUS_EXCLUDED, "contradicting", start))
        else:
            candidates.append((item, start))

    # 3: dedupe and cap, per (dimension, indicator), in a fixed order.
    # Order within a group: earliest in the transcript first; for the same
    # start, the longer quote first (so the shorter one is the duplicate);
    # then stronger first, so an identical quote tagged twice keeps the
    # stronger tag. Nothing here reads confidence.
    def _group_order(c: tuple[EvidenceItem, int]):
        item, start = c
        return (start, -len(normalize_text(item.quote)), normalize_text(item.quote),
                -float(strengths[item.strength]["weight"]), item.strength)

    groups: dict[tuple[str, str], list[tuple[EvidenceItem, int]]] = {}
    for c in candidates:
        groups.setdefault((c[0].dimension, c[0].indicator), []).append(c)

    accepted: list[tuple[EvidenceItem, int]] = []
    for key in sorted(groups):
        kept: list[tuple[EvidenceItem, int]] = []
        kept_quotes: list[str] = []
        ordered = sorted(groups[key], key=_group_order)
        # Longest first, so a quote is only ever compared against longer-or-equal kept ones.
        for item, start in sorted(ordered, key=lambda c: -len(normalize_text(c[0].quote))):
            q = normalize_text(item.quote)
            if any(q in k for k in kept_quotes):
                records.append((item, STATUS_EXCLUDED, "duplicate", start))
            else:
                kept.append((item, start))
                kept_quotes.append(q)
        # TODO(team-decision): which items survive the cap. Default: the
        # earliest in the transcript (not the strongest), so the cap can't
        # be used to cherry-pick high-weight evidence. Decision doc §6
        # decided the cap of 2, not the selection rule.
        kept.sort(key=_group_order)
        for item, start in kept[:MAX_ITEMS_PER_INDICATOR]:
            accepted.append((item, start))
        for item, start in kept[MAX_ITEMS_PER_INDICATOR:]:
            records.append((item, STATUS_EXCLUDED, "over_cap", start))

    # 4 + 5: map to SignalUnits in the deterministic order.
    def _signal_order(c: tuple[EvidenceItem, int]):
        item, start = c
        return (item.dimension, start, item.indicator, normalize_text(item.quote))

    accepted.sort(key=_signal_order)
    signals: list[SignalUnit] = []
    decisions: list[EvidenceDecision] = []
    for item, start in accepted:
        ref = _source_ref(item, start)
        signals.append(SignalUnit(
            trait=item.dimension,
            value=item.indicator,
            weight=float(strengths[item.strength]["weight"]),
            source_ref=ref,
        ))
        decisions.append(EvidenceDecision(
            item=item, status=STATUS_ACCEPTED, reason=None,
            needs_review=_needs_review(STATUS_ACCEPTED, None, item, floors, confidence_review_threshold),
            quote_start=start, source_ref=ref,
        ))

    for item, status, reason, start in records:
        decisions.append(EvidenceDecision(
            item=item, status=status, reason=reason,
            needs_review=_needs_review(status, reason, item, floors, confidence_review_threshold),
            quote_start=start,
        ))

    decisions.sort(key=_decision_order)
    return signals, decisions


def _decision_order(d: EvidenceDecision):
    it = d.item
    return (
        str(it.dimension),
        d.quote_start is None, d.quote_start if d.quote_start is not None else 0,
        str(it.indicator),
        normalize_text(it.quote) if isinstance(it.quote, str) else "",
        str(it.polarity),
        str(it.strength),
    )
