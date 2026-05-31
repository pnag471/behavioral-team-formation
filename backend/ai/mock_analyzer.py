"""
Keyword-based mock implementation of BehavioralAnalyzer.
Replace this class with an LLM-backed implementation to upgrade the prototype.
See interfaces.py for the contract.
"""
from __future__ import annotations
from typing import List, TYPE_CHECKING

from ai.interfaces import BehavioralAnalyzer

if TYPE_CHECKING:
    from app.models import AssessmentRequest

# ---------------------------------------------------------------------------
# Keyword maps — each dimension maps category → trigger words
# ---------------------------------------------------------------------------
_CONFLICT = {
    "collaborative":   ["discuss", "together", "understand", "listen", "check in", "talk", "share",
                        "empathize", "approach", "address", "meet", "open", "honest"],
    "confrontational": ["confront", "call out", "address directly", "tell them", "challenge",
                        "push back", "demand", "escalate", "raise", "assert"],
    "avoidant":        ["avoid", "ignore", "wait", "hope", "maybe later", "not my place",
                        "uncomfortable", "stay out", "move on", "drop"],
}

_LEADERSHIP = {
    "directive":    ["take charge", "take over", "assign", "decide", "lead", "step up",
                     "organize everyone", "set direction", "make the call", "delegate"],
    "facilitative": ["coordinate", "facilitate", "organize", "structure", "schedule", "guide",
                     "help the team", "bring together", "clarify", "mediate"],
    "emergent":     ["see what happens", "wait for", "let someone else", "if needed",
                     "naturally", "whoever", "depends", "open to"],
}

_PLANNING = {
    "planner":      ["plan", "schedule", "timeline", "deadline", "milestone", "roadmap",
                     "structure", "breakdown", "outline", "gantt", "sprint"],
    "spontaneous":  ["flexible", "see how it goes", "figure out", "play it by ear",
                     "go with the flow", "improvise", "adapt as we go"],
    "adaptive":     ["adjust", "check in", "review", "iterate", "reassess", "update",
                     "revisit", "course correct"],
}

_COMMUNICATION = {
    "async":  ["document", "write", "message", "slack", "async", "note", "log",
               "update the wiki", "comment", "record", "post"],
    "sync":   ["meeting", "call", "standup", "video", "sync", "talk", "discuss live",
               "in person", "zoom", "hangout"],
    "mixed":  ["both", "depends", "sometimes async", "mix", "either", "combination"],
}

_ACCOUNTABILITY = {
    "high":   ["commit", "own", "responsible", "accountable", "follow through",
               "deliver", "promise", "ensure", "make sure", "on time"],
    "medium": ["try", "do my best", "hope", "expect", "should", "attempt"],
    "low":    ["depends", "hard to say", "not always", "if possible", "busy"],
}

_HELP_SEEKING = {
    "proactive": ["ask", "reach out", "check in with", "look for", "request help",
                  "seek feedback", "proactively", "early", "before"],
    "reactive":  ["when stuck", "if needed", "after trying", "once i", "only if",
                  "last resort", "when i have to"],
    "independent": ["figure it out", "on my own", "self-sufficient", "research first",
                    "independently", "without help", "prefer to solve"],
}

_CONFIDENCE_BOOSTERS = [
    "confident", "certain", "definitely", "absolutely", "strong", "expert",
    "experienced", "clear", "sure", "know how", "always", "will",
]

_EXECUTION = {
    "methodical":  ["step by step", "systematic", "process", "checklist", "structured",
                    "careful", "thorough", "deliberate"],
    "iterative":   ["iterate", "refine", "improve", "feedback loop", "incrementally",
                    "build and test", "prototype"],
    "exploratory": ["explore", "experiment", "try different", "discover", "investigate",
                    "open ended", "creative", "research"],
}


def _score(text: str, keyword_map: dict[str, list[str]]) -> str:
    """Return the category with the most keyword hits in text."""
    lower = text.lower()
    scores = {cat: sum(1 for kw in kws if kw in lower) for cat, kws in keyword_map.items()}
    best = max(scores, key=scores.get)
    return best


def _score_all(responses: list[dict]) -> str:
    """Concatenate all response texts for bulk scoring."""
    return " ".join(r.get("response_text", "") for r in responses)


class MockBehavioralAnalyzer(BehavioralAnalyzer):
    def analyze(self, request: "AssessmentRequest") -> dict:
        raw = [r.model_dump() for r in request.responses]
        combined = _score_all(raw)

        # Boost signals from option choices
        option_text = " ".join(r.get("option_id", "") for r in raw)
        full_text = combined + " " + option_text

        confidence_boost = sum(
            0.05 for booster in _CONFIDENCE_BOOSTERS if booster in full_text.lower()
        )
        confidence_score = round(min(0.9, max(0.3, 0.45 + confidence_boost)), 2)

        return {
            "planning_style":       _score(full_text, _PLANNING),
            "communication_style":  _score(full_text, _COMMUNICATION),
            "execution_style":      _score(full_text, _EXECUTION),
            "conflict_style":       _score(full_text, _CONFLICT),
            "leadership_style":     _score(full_text, _LEADERSHIP),
            "accountability":       _score(full_text, _ACCOUNTABILITY),
            "help_seeking":         _score(full_text, _HELP_SEEKING),
            "confidence_score":     confidence_score,
        }
