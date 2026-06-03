from __future__ import annotations
import json
import os
from typing import TYPE_CHECKING

import anthropic

from ai.interfaces import BehavioralAnalyzer

if TYPE_CHECKING:
    from app.models import AssessmentRequest


class ClaudeBehavioralAnalyzer(BehavioralAnalyzer):

    def __init__(self):
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        prompt_path = os.path.join(os.path.dirname(__file__), "..", "..", "prompts", "behavioral_assessment.txt")        
        with open(prompt_path) as f:
            self.prompt_template = f.read()

    def analyze(self, request: "AssessmentRequest") -> dict:
        # Format responses for the prompt
        responses_text = ""
        for i, response in enumerate(request.responses, 1):
            responses_text += f"\nScenario {i}:\n"
            responses_text += f"  Option chosen: {response.option_id}\n"
            if response.response_text:
                responses_text += f"  Written response: {response.response_text}\n"

        # Fill in the prompt template
        prompt = self.prompt_template.replace("{responses}", responses_text)

        # Call Claude
        message = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        # Parse the JSON response
        raw = message.content[0].text
        result = json.loads(raw)
        return result