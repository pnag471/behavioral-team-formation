from __future__ import annotations
import json
import os
from typing import List, TYPE_CHECKING

import anthropic

from ai.interfaces import ExplanationGenerator, NormGenerator

if TYPE_CHECKING:
    from app.models import Student, Team, TeamExplanation, TeamNorm


class ClaudeExplainer(ExplanationGenerator, NormGenerator):

    def __init__(self):
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        prompt_path = os.path.join(os.path.dirname(__file__), "..", "..", "prompts", "explanation_generation.txt")
        with open(prompt_path) as f:
            self.prompt_template = f.read()

    def suggest_norms(self, team: "Team", students: "List[Student]") -> "List[TeamNorm]":
        # Norms are generated as part of the explanation call
        # so we generate the full explanation and return just the norms
        explanation = self.generate(team, students)
        return explanation.team_norms

    def generate(self, team: "Team", students: "List[Student]") -> "TeamExplanation":
        from app.models import TeamExplanation, TeamNorm

        sb = team.score_breakdown

        # Format team members for the prompt
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

        # Format norms already generated
        norms_data = [
            {"category": n.category, "norm": n.norm}
            for n in team.team_norms
        ]

        # Fill in the prompt template
        prompt = self.prompt_template \
            .replace("{team_members_json}", json.dumps(members_data, indent=2)) \
            .replace("{total_score}", f"{sb.total:.0%}") \
            .replace("{skill_coverage}", f"{sb.skill_coverage:.0%}") \
            .replace("{behavioral_compat}", f"{sb.behavioral_compat:.0%}") \
            .replace("{availability_overlap}", f"{sb.availability_overlap:.0%}") \
            .replace("{conflict_risk}", f"{sb.conflict_risk:.0%}") \
            .replace("{match_confidence}", f"{sb.match_confidence:.0%}") \
            .replace("{team_norms_json}", json.dumps(norms_data, indent=2))

        # Call Claude
        message = self.client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        # Parse response
        raw = message.content[0].text
        result = json.loads(raw)

        return TeamExplanation(
            team_id=team.id,
            compatibility_score=sb.total,
            match_confidence=sb.match_confidence,
            strengths=result["strengths"],
            risks=result["risks"],
            explanation=result["explanation"],
            team_norms=team.team_norms,
        )