"""
Abstract interfaces for AI-powered components.

To swap in a real LLM (e.g., Claude via the Anthropic SDK):
1. Create a new class that inherits from BehavioralAnalyzer / ExplanationGenerator / NormGenerator
2. Implement the abstract methods using your LLM client
3. Instantiate the new class in behavioral_analysis.py / explainability.py

No other code changes are required.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from app.models import AssessmentRequest, Student, Team, TeamExplanation, TeamNorm


class BehavioralAnalyzer(ABC):
    @abstractmethod
    def analyze(self, request: "AssessmentRequest") -> dict:
        """
        Infer behavioral signature from scenario responses.

        Returns a dict with keys:
          planning_style, communication_style, execution_style,
          conflict_style, leadership_style, accountability, help_seeking,
          confidence_score
        """
        ...


class ExplanationGenerator(ABC):
    @abstractmethod
    def generate(self, team: "Team", students: List["Student"]) -> "TeamExplanation":
        """Generate natural-language explanation for a team."""
        ...


class NormGenerator(ABC):
    @abstractmethod
    def suggest_norms(self, team: "Team", students: List["Student"]) -> List["TeamNorm"]:
        """Generate suggested working norms for a team based on behavioral signatures."""
        ...
