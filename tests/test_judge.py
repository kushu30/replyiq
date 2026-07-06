import json
import os
from unittest.mock import MagicMock, patch

import pytest

os.environ.setdefault("GROQ_API_KEY", "test-key-not-real")

from evaluation.judge import LlmJudgeMetric, JUDGE_CRITERIA


def _fake_groq_response(payload: dict):
    fake_message = MagicMock()
    fake_message.content = json.dumps(payload)
    fake_choice = MagicMock()
    fake_choice.message = fake_message
    fake_response = MagicMock()
    fake_response.choices = [fake_choice]
    return fake_response


def test_parse_response_handles_valid_json():
    judge = LlmJudgeMetric()
    payload = {"correctness": 8, "helpfulness": 7, "completeness": 8, "professionalism": 9, "tone": 9, "groundedness": 6, "reasoning": "solid reply"}
    parsed = judge._parse_response(json.dumps(payload))
    assert parsed["correctness"] == 8
    assert parsed["groundedness"] == 6


def test_parse_response_handles_malformed_json_gracefully():
    judge = LlmJudgeMetric()
    parsed = judge._parse_response("not valid json at all")
    assert all(parsed[criterion] == 0 for criterion in JUDGE_CRITERIA)
    assert "reasoning" in parsed


def test_parse_response_strips_markdown_fences():
    judge = LlmJudgeMetric()
    payload = {"correctness": 10, "helpfulness": 10, "completeness": 10, "professionalism": 10, "tone": 10, "groundedness": 10, "reasoning": "great"}
    wrapped = f"```json\n{json.dumps(payload)}\n```"
    parsed = judge._parse_response(wrapped)
    assert parsed["correctness"] == 10


def test_score_averages_criteria_correctly():
    judge = LlmJudgeMetric()
    payload = {"correctness": 10, "helpfulness": 10, "completeness": 10, "professionalism": 10, "tone": 10, "groundedness": 0, "reasoning": "mixed"}

    with patch.object(judge.client.chat.completions, "create", return_value=_fake_groq_response(payload)):
        result = judge.score(generated="test reply", reference="test reference", customer_email="test email")

    expected_average = (10 + 10 + 10 + 10 + 10 + 0) / 6 / 10
    assert result.score == round(expected_average, 4)


def test_score_works_without_reference():
    judge = LlmJudgeMetric()
    payload = {"correctness": 7, "helpfulness": 7, "completeness": 7, "professionalism": 7, "tone": 7, "groundedness": 7, "reasoning": "ok"}

    with patch.object(judge.client.chat.completions, "create", return_value=_fake_groq_response(payload)):
        result = judge.score(generated="test reply", customer_email="test email")

    assert result.score == 0.7