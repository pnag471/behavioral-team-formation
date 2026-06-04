from __future__ import annotations
import json
import os
from typing import List, TYPE_CHECKING

try:
    import anthropic
    _ANTHROPIC_AVAILABLE = True
except ImportError:
    _ANTHROPIC_AVAILABLE = False

from ai.interfaces import ExplanationGenerator, NormGenerator

if TYPE_CHECKING:
    from app.models import Student, Team, TeamExplanation, TeamNorm


class ClaudeExplainer(ExplanationGenerator, NormGenerator):

    def __init__(self):
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if _ANTHROPIC_AVAILABLE and api_key:
            self.client = anthropic.Anthropic(api_key=api_key)
            prompt_path = os.path.join(
                os.path.dirname(__file__), "..", "..", "prompts", "explanation_generation.txt"
            )
            with open(prompt_path) as f:
                self.prompt_template = f.read()
            self._fallback = None
        else:
            from ai.mock_explainer import MockExplainer
            self._fallback = MockExplainer()
            self.client = None
            self.prompt_template = None

    def suggest_norms(self, team: "Team", students: "List[Student]") -> "List[TeamNorm]":
        if self._fallback is not None:
            return self._fallback.suggest_norms(team, students)
        return self.generate(team, students).team_norms

    def generate(self, team: "Team", students: "List[Student]") -> "TeamExplanation":
        if self._fallback is not None:
            return self._fallback.generate(team, students)

        from app.models import TeamExplanation

        sb = team.score_breakdown

        members_data = [
            {
                "name": s.name,
                "skills": s.competence_signature.skills,
                "roles": s.competence_signature.roles,
                "experience_level": s.competence_signature.experience_level,
                "planning_style": s.work_rhythm_signature.planning_style,
                "communication_style": s.work_rhythm_signature.communication_style,
                "execution_style": s.work_rhythm_signature.execution_style,
                "conflict_style": s.collaboration_signature.conflict_style,
                "leadership_style": s.collaboration_signature.leadership_style,
                "accountability": s.collaboration_signature.accountability,
                "help_seeking": s.collaboration_signature.help_seeking,
            }
            for s in students
        ]

        norms_data = [{"category": n.category, "norm": n.norm} for n in team.team_norms]

        prompt = (
            self.prompt_template
            .replace("{team_members_json}", json.dumps(members_data, indent=2))
            .replace("{total_score}", f"{sb.total:.0%}")
            .replace("{skill_coverage}", f"{sb.skill_coverage:.0%}")
            .replace("{behavioral_compat}", f"{sb.behavioral_compat:.0%}")
            .replace("{availability_overlap}", f"{sb.availability_overlap:.0%}")
            .replace("{conflict_risk}", f"{sb.conflict_risk:.0%}")
            .replace("{match_confidence}", f"{sb.match_confidence:.0%}")
            .replace("{team_norms_json}", json.dumps(norms_data, indent=2))
        )

        message = self.client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )

        result = json.loads(message.content[0].text)

        return TeamExplanation(
            team_id=team.id,
            compatibility_score=sb.total,
            match_confidence=sb.match_confidence,
            strengths=result["strengths"],
            risks=result["risks"],
            explanation=result["explanation"],
            team_norms=team.team_norms,
        )
