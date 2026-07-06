import json
import os
import re

from dotenv import load_dotenv
from groq import Groq

from generator.rate_limit import retry_with_backoff

load_dotenv()


class RetrievalEvaluator:
    def __init__(self, model_name: str = "llama-3.1-8b-instant") -> None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError("GROQ_API_KEY not found in environment")
        self.client = Groq(api_key=api_key)
        self.model_name = model_name

    def domain_precision(self, query_domain: str, retrieved_examples: list[dict]) -> float:
        if not retrieved_examples:
            return 0.0
        matches = sum(1 for example in retrieved_examples if example.get("domain") == query_domain)
        return round(matches / len(retrieved_examples), 4)

    def confidence_score(self, retrieved_examples: list[dict]) -> float:
        if not retrieved_examples:
            return 0.0
        similarities = [example.get("similarity_score", 0.0) for example in retrieved_examples]
        return round(sum(similarities) / len(similarities), 4)

    def _build_relevance_prompt(self, customer_email: str, retrieved_examples: list[dict]) -> str:
        examples_block = "\n\n".join(
            f"Retrieved example {i + 1} (domain: {example['domain']}):\n"
            f"Customer: {example['customer_email']}\n"
            f"Reply: {example['support_reply']}"
            for i, example in enumerate(retrieved_examples)
        )

        return f"""You are evaluating a retrieval system used for grounding an AI support reply.

New customer email to be answered:
{customer_email}

Retrieved examples intended to provide relevant context for answering it:
{examples_block}

Judge ONLY the retrieval quality — not any reply. For each retrieved example, decide
if it is actually relevant and useful context for answering the new customer email
(same underlying issue or a closely related one), or irrelevant noise.

Respond with ONLY valid JSON in this exact format, no markdown, no explanation:
{{"relevant_count": <int, out of {len(retrieved_examples)}>, "relevance_score": <int 0-10, overall usefulness of the retrieved set>, "reasoning": "<one sentence explaining any irrelevant example>"}}"""

    def _parse_response(self, raw_text: str) -> dict:
        cleaned = re.sub(r"```json|```", "", raw_text).strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return {"relevant_count": 0, "relevance_score": 0, "reasoning": "Failed to parse retrieval judge response"}

    @retry_with_backoff()
    def judge_relevance(self, customer_email: str, retrieved_examples: list[dict]) -> dict:
        prompt = self._build_relevance_prompt(customer_email, retrieved_examples)
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
        )
        return self._parse_response(response.choices[0].message.content)