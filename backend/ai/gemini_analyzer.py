from __future__ import annotations
import json
import os
from typing import TYPE_CHECKING

from google import genai
from google.genai import types

from ai.interfaces import BehavioralAnalyzer

if TYPE_CHECKING:
    from app.models import AssessmentRequest

class GeminiBehavioralAnalyzer(BehavioralAnalyzer):

    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            self.client = genai.Client(api_key=api_key)
            prompt_path = os.path.join(
                os.path.dirname(__file__), "..", "..", "prompts", "behavioral_assessment.txt"
            )
            with open(prompt_path) as f:
                self.prompt_template = f.read()
            self._fallback = None
        else:
            from ai.mock_analyzer import MockBehavioralAnalyzer
            self._fallback = MockBehavioralAnalyzer()
            self.model = None
            self.prompt_template = None

    def analyze(self, request: "AssessmentRequest") -> dict:
        if self._fallback is not None:
            return self._fallback.analyze(request)

        # Format responses
        responses_text = ""
        for i, response in enumerate(request.responses, 1):
            responses_text += f"\nScenario {i}:\n"
            responses_text += f"  Option chosen: {response.option_id}\n"
            if response.response_text:
                responses_text += f"  Written response: {response.response_text}\n"

        # Fill prompt template
        prompt = self.prompt_template.replace("{responses}", responses_text)

        # Call Gemini
        response = self.client.models.generate_content(
            model="gemini-1.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.2,
                response_mime_type="application/json",
            )
        )

        result = json.loads(response.text)
        return result
        