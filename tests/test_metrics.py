from evaluation.metrics import BleuMetric, RougeLMetric, WeightedEvaluator


def test_bleu_identical_text_scores_high():
    metric = BleuMetric()
    result = metric.score("thank you for reaching out", "thank you for reaching out")
    assert result.score > 0.9


def test_bleu_unrelated_text_scores_low():
    metric = BleuMetric()
    result = metric.score("thank you for reaching out", "the weather today is sunny and warm")
    assert result.score < 0.2


def test_rouge_l_identical_text_scores_high():
    metric = RougeLMetric()
    result = metric.score("your refund has been processed", "your refund has been processed")
    assert result.score > 0.9


def test_rouge_l_partial_overlap_scores_between_bounds():
    metric = RougeLMetric()
    result = metric.score("your refund has been processed today", "your refund was processed")
    assert 0.0 < result.score < 1.0


def test_weighted_evaluator_combines_only_provided_metrics():
    evaluator = WeightedEvaluator(
        metrics=[BleuMetric(), RougeLMetric()],
        weights={"bleu": 0.5, "rouge_l": 0.5},
    )
    result = evaluator.evaluate("thank you", "thank you")
    assert "bleu" in result["components"]
    assert "rouge_l" in result["components"]
    assert 0.0 <= result["overall_score"] <= 1.0


def test_weighted_evaluator_missing_weight_defaults_to_zero_contribution():
    evaluator = WeightedEvaluator(metrics=[BleuMetric()], weights={})
    result = evaluator.evaluate("thank you", "thank you")
    assert result["overall_score"] == 0.0