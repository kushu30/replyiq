import json
import os
from unittest.mock import MagicMock, patch

os.environ.setdefault("GROQ_API_KEY", "test-key-not-real")

from evaluation.retrieval_evaluator import RetrievalEvaluator


def _fake_groq_response(payload: dict):
    fake_message = MagicMock()
    fake_message.content = json.dumps(payload)
    fake_choice = MagicMock()
    fake_choice.message = fake_message
    fake_response = MagicMock()
    fake_response.choices = [fake_choice]
    return fake_response


def test_domain_precision_all_matching():
    evaluator = RetrievalEvaluator()
    retrieved = [{"domain": "refunds"}, {"domain": "refunds"}, {"domain": "refunds"}]
    assert evaluator.domain_precision("refunds", retrieved) == 1.0


def test_domain_precision_partial_match():
    evaluator = RetrievalEvaluator()
    retrieved = [{"domain": "refunds"}, {"domain": "billing"}, {"domain": "refunds"}]
    assert evaluator.domain_precision("refunds", retrieved) == round(2 / 3, 4)


def test_domain_precision_no_match():
    evaluator = RetrievalEvaluator()
    retrieved = [{"domain": "billing"}, {"domain": "login_issues"}]
    assert evaluator.domain_precision("refunds", retrieved) == 0.0


def test_domain_precision_empty_retrieval():
    evaluator = RetrievalEvaluator()
    assert evaluator.domain_precision("refunds", []) == 0.0


def test_judge_relevance_parses_mocked_response():
    evaluator = RetrievalEvaluator()
    payload = {"relevant_count": 2, "relevance_score": 7, "reasoning": "one example was off-topic"}
    retrieved = [
        {"domain": "refunds", "customer_email": "broken item", "support_reply": "refunded"},
        {"domain": "billing", "customer_email": "double charge", "support_reply": "fixed"},
    ]

    with patch.object(evaluator.client.chat.completions, "create", return_value=_fake_groq_response(payload)):
        result = evaluator.judge_relevance("my order arrived broken", retrieved)

    assert result["relevance_score"] == 7
    assert result["relevant_count"] == 2


def test_judge_relevance_handles_malformed_response():
    evaluator = RetrievalEvaluator()
    fake_message = MagicMock()
    fake_message.content = "not json"
    fake_choice = MagicMock()
    fake_choice.message = fake_message
    fake_response = MagicMock()
    fake_response.choices = [fake_choice]

    with patch.object(evaluator.client.chat.completions, "create", return_value=fake_response):
        result = evaluator.judge_relevance("test email", [{"domain": "refunds", "customer_email": "x", "support_reply": "y"}])

    assert result["relevance_score"] == 0