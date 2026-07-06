import os

import google.generativeai as genai
from dotenv import load_dotenv

from generator.rate_limit import retry_with_backoff

load_dotenv()


class ReplyGenerator:
    def __init__(self, model_name: str = "gemini-2.5-flash-lite") -> None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError("GEMINI_API_KEY not found in environment")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)

    def _build_prompt(self, customer_email: str, similar_examples: list[dict]) -> str:
        examples_block = "\n\n".join(
            f"Example {i + 1}:\n"
            f"Customer: {example['customer_email']}\n"
            f"Support Reply: {example['support_reply']}"
            for i, example in enumerate(similar_examples)
        )

        return f"""You are a customer support agent writing an email reply.

Use the following historical examples as style and tone references. Do not copy them verbatim, adapt the approach to the new customer's specific issue.

{examples_block}

New customer email:
{customer_email}

Write a professional, concise, and empathetic support reply. Only output the reply text, with no preamble, no subject line, and no explanation.

Do not invent specific facts that are not stated in the customer's email or the examples above — no fabricated causes (e.g. claiming a package was "misrouted" or "held at a facility" if the customer never said that and no example confirms it), no invented order details, no specific dates or timelines you cannot actually confirm. If the cause or status is unknown, say you're looking into it rather than asserting a specific explanation."""

    @retry_with_backoff()
    def generate_reply(self, customer_email: str, similar_examples: list[dict]) -> str:
        prompt = self._build_prompt(customer_email, similar_examples)
        response = self.model.generate_content(prompt)
        return response.text.strip()