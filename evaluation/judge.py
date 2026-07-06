import json
import os
import re

from dotenv import load_dotenv
from groq import Groq

from evaluation.metrics import EvaluationMetric, MetricResult
from generator.rate_limit import retry_with_backoff

load_dotenv()

JUDGE_CRITERIA = ["correctness", "helpfulness", "completeness", "professionalism", "tone", "groundedness"]


class LlmJudgeMetric(EvaluationMetric):
    def __init__(self, model_name: str = "llama-3.3-70b-versatile") -> None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError("GROQ_API_KEY not found in environment")
        self.client = Groq(api_key=api_key)
        self.model_name = model_name

    def _build_prompt(self, customer_email: str, generated_reply: str, reference_reply: str) -> str:
        reference_block = (
            f"Reference reply (written by a human agent, for comparison only):\n{reference_reply}\n"
            if reference_reply.strip()
            else "No reference reply is available. Judge the AI reply purely on its own merits "
            "against general customer support best practices, not by comparison.\n"
        )

        return f"""You are evaluating a customer support reply written by an AI system.

Customer email:
{customer_email}

AI-generated reply:
{generated_reply}

{reference_block}
Rate the AI-generated reply from 0 to 10 on each of these dimensions. Be a strict,
skeptical evaluator — a real support quality reviewer, not a cheerleader. Most
replies have at least one real flaw; find it. Reserve scores of 9-10 only for
replies with no meaningful weakness at all. A generic, templated-sounding reply
that never references specifics from the customer's actual email should score
no higher than 6-7 on completeness and helpfulness, even if it's polite and well
structured.
- correctness: does it address the customer's actual issue accurately
- helpfulness: does it move the customer toward resolution
- completeness: does it cover everything the customer asked about
- professionalism: is the language appropriate for customer support
- tone: is it empathetic and suitably toned for the situation
- groundedness: does it avoid inventing specific facts not present in the customer's
  email (fabricated causes, invented order details, unconfirmed timelines)? A reply
  that confidently states an unverified cause (e.g. claiming a package was
  "misrouted" when the customer never said so and nothing confirms it) must score
  low here (0-3), even if it sounds helpful and well-written.

Respond with ONLY valid JSON in this exact format, no markdown, no explanation:
{{"correctness": <int>, "helpfulness": <int>, "completeness": <int>, "professionalism": <int>, "tone": <int>, "groundedness": <int>, "weakness": "<the single most significant flaw in this reply, be specific>", "reasoning": "<one sentence summary>"}}"""

    def _parse_response(self, raw_text: str) -> dict:
        cleaned = re.sub(r"```json|```", "", raw_text).strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return {criterion: 0 for criterion in JUDGE_CRITERIA} | {
                "reasoning": "Failed to parse judge response"
            }

    @retry_with_backoff()
    def score(self, generated: str, reference: str = "", customer_email: str = "") -> MetricResult:
        prompt = self._build_prompt(customer_email, generated, reference)
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
        )
        parsed = self._parse_response(response.choices[0].message.content)

        criterion_scores = [parsed.get(criterion, 0) for criterion in JUDGE_CRITERIA]
        average_score = sum(criterion_scores) / len(criterion_scores) / 10

        detail = json.dumps(parsed)
        return MetricResult(name="llm_judge", score=round(average_score, 4), detail=detail)