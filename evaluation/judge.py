import json
import os
import re

import google.generativeai as genai
from dotenv import load_dotenv

from evaluation.metrics import EvaluationMetric, MetricResult
from generator.rate_limit import retry_with_backoff

load_dotenv()

JUDGE_CRITERIA = ["correctness", "helpfulness", "completeness", "professionalism", "tone", "groundedness"]


class LlmJudgeMetric(EvaluationMetric):
    def __init__(self, model_name: str = "gemini-2.5-flash-lite") -> None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError("GEMINI_API_KEY not found in environment")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)

    def _build_prompt(self, customer_email: str, generated_reply: str, reference_reply: str) -> str:
        return f"""You are evaluating a customer support reply written by an AI system.

Customer email:
{customer_email}

AI-generated reply:
{generated_reply}

Reference reply (written by a human agent, for comparison only):
{reference_reply}

Rate the AI-generated reply from 0 to 10 on each of these dimensions:
- correctness: does it address the customer's actual issue accurately
- helpfulness: does it move the customer toward resolution
- completeness: does it cover everything the customer asked about
- professionalism: is the language appropriate for customer support
- tone: is it empathetic and suitably toned for the situation
- groundedness: does it avoid hallucinating specific facts not present in the customer email or instructions

Respond with ONLY valid JSON in this exact format, no markdown, no explanation:
{{"correctness": <int>, "helpfulness": <int>, "completeness": <int>, "professionalism": <int>, "tone": <int>, "groundedness": <int>, "reasoning": "<one sentence summary>"}}"""

    def _parse_response(self, raw_text: str) -> dict:
        cleaned = re.sub(r"```json|```", "", raw_text).strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return {criterion: 0 for criterion in JUDGE_CRITERIA} | {
                "reasoning": "Failed to parse judge response"
            }

    @retry_with_backoff()
    def score(self, generated: str, reference: str, customer_email: str = "") -> MetricResult:
        prompt = self._build_prompt(customer_email, generated, reference)
        response = self.model.generate_content(prompt)
        parsed = self._parse_response(response.text)

        criterion_scores = [parsed.get(criterion, 0) for criterion in JUDGE_CRITERIA]
        average_score = sum(criterion_scores) / len(criterion_scores) / 10

        detail = json.dumps(parsed)
        return MetricResult(name="llm_judge", score=round(average_score, 4), detail=detail)