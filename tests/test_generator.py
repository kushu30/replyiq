import os

os.environ.setdefault("GROQ_API_KEY", "test-key-not-real")

from generator.generator import ReplyGenerator


def test_normal_prompt_includes_retrieved_examples():
    generator = ReplyGenerator()
    examples = [{"customer_email": "my item broke", "support_reply": "refund issued"}]
    prompt = generator._build_prompt("my item is also broken", examples, low_confidence=False)
    assert "my item broke" in prompt
    assert "refund issued" in prompt


def test_normal_prompt_forbids_fabrication():
    generator = ReplyGenerator()
    prompt = generator._build_prompt("test email", [], low_confidence=False)
    assert "fabricated" in prompt.lower() or "invent" in prompt.lower()


def test_low_confidence_prompt_excludes_examples_and_hedges():
    generator = ReplyGenerator()
    examples = [{"customer_email": "unrelated topic", "support_reply": "unrelated reply"}]
    prompt = generator._build_prompt("test email", examples, low_confidence=True)
    assert "unrelated topic" not in prompt
    assert "don't have enough information" in prompt.lower() or "uncertain" in prompt.lower() or "looking into" in prompt.lower()


def test_low_confidence_prompt_still_includes_customer_email():
    generator = ReplyGenerator()
    prompt = generator._build_prompt("specific customer complaint text", [], low_confidence=True)
    assert "specific customer complaint text" in prompt