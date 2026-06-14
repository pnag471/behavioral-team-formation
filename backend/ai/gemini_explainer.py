from __future__ import annotations
import json
import os
from typing import List, TYPE_CHECKING

from google import genai
from google.genai import types

from ai.interfaces import ExplanationGenerator, NormGenerator

if TYPE_CHECKING:
    from app.models import Student, Team, TeamExplanation, TeamNorm


class GeminiExplainer(ExplanationGenerator, NormGenerator):

    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            self.client = genai.Client(api_key=api_key)
            prompt_path = os.path.join(
                os.path.dirname(__file__), "..", "..", "prompts", "explanation_generation.txt"
            )
            with open(prompt_path) as f:
                self.prompt_template = f.read()
            self._fallback = None
        else:
            from ai.mock_explainer import MockExplainer
            self._fallback = MockExplainer()
            self.model = None
            self.prompt_template = None

    def suggest_norms(self, team: "Team", students: "List[Student]") -> "List[TeamNorm]":
        explanation = self.generate(team, students)
        return explanation.team_norms

    def generate(self, team: "Team", students: "List[Student]") -> "TeamExplanation":
        if self._fallback is not None:
            return self._fallback.generate(team, students)

        from app.models import TeamExplanation

        sb = team.score_breakdown

        members_data = []
        for student in students:
            members_data.append({
                "name": student.name,
                "skills": student.competence_signature.skills,
                "roles": student.competence_signature.roles,
                "experience_level": student.competence_signature.experience_level,
                "planning_style": student.work_rhythm_signature.planning_style,
                "communication_style": student.work_rhythm_signature.communication_style,
                "execution_style": student.work_rhythm_signature.execution_style,
                "conflict_style": student.collaboration_signature.conflict_style,
                "leadership_style": student.collaboration_signature.leadership_style,
                "accountability": student.collaboration_signature.accountability,
                "help_seeking": student.collaboration_signature.help_seeking,
            })

        norms_data = [
            {"category": n.category, "norm": n.norm}
            for n in team.team_norms
        ]

        prompt = self.prompt_template \
            .replace("{team_members_json}", json.dumps(members_data, indent=2)) \
            .replace("{total_score}", f"{sb.total:.0%}") \
            .replace("{skill_coverage}", f"{sb.skill_coverage:.0%}") \
            .replace("{behavioral_compat}", f"{sb.behavioral_compat:.0%}") \
            .replace("{availability_overlap}", f"{sb.availability_overlap:.0%}") \
            .replace("{conflict_risk}", f"{sb.conflict_risk:.0%}") \
            .replace("{match_confidence}", f"{sb.match_confidence:.0%}") \
            .replace("{team_norms_json}", json.dumps(norms_data, indent=2))

        response = self.client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.2,
                response_mime_type="application/json",
            )
        )
        result = json.loads(response.text)

        return TeamExplanation(
            team_id=team.id,
            compatibility_score=sb.total,
            match_confidence=sb.match_confidence,
            strengths=result["strengths"],
            risks=result["risks"],
            explanation=result["explanation"],
            team_norms=team.team_norms,
        )