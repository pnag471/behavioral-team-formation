from __future__ import annotations
import json
import os
from typing import TYPE_CHECKING

try:
    import anthropic
    _ANTHROPIC_AVAILABLE = True
except ImportError:
    _ANTHROPIC_AVAILABLE = False

from ai.interfaces import BehavioralAnalyzer

if TYPE_CHECKING:
    from app.models import AssessmentRequest

MODEL_ID = "claude-sonnet-4-6"


class ClaudeBehavioralAnalyzer(BehavioralAnalyzer):

    def __init__(self):
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if _ANTHROPIC_AVAILABLE and api_key:
            self.client = anthropic.Anthropic(api_key=api_key)
            prompt_path = os.path.join(
                os.path.dirname(__file__), "..", "..", "prompts", "behavioral_assessment.txt"
            )
            with open(prompt_path) as f:
                raw = f.read()
            # Split at {responses}: static prefix → cached system message
            # dynamic suffix → appended to the user turn after the responses
            parts = raw.split("{responses}")
            self._static_prefix = parts[0]
            self._dynamic_suffix = parts[1] if len(parts) > 1 else ""
            self._fallback = None
        else:
            from ai.mock_analyzer import MockBehavioralAnalyzer
            self._fallback = MockBehavioralAnalyzer()
            self.client = None
            self._static_prefix = None
            self._dynamic_suffix = None

    def analyze(self, request: "AssessmentRequest") -> dict:
        if self._fallback is not None:
            return self._fallback.analyze(request)

        responses_text = ""
        for i, response in enumerate(request.responses, 1):
            responses_text += f"\nScenario {i}:\n"
            responses_text += f"  Option chosen: {response.option_id}\n"
            if response.response_text:
                responses_text += f"  Written response: {response.response_text}\n"

        user_content = responses_text + self._dynamic_suffix

        message = self.client.messages.create(
            model=MODEL_ID,
            max_tokens=1024,
            system=[
                {
                    "type": "text",
                    "text": self._static_prefix,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": user_content}],
        )

        return json.loads(message.content[0].text)
