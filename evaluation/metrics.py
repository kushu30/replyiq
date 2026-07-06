from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class MetricResult:
    name: str
    score: float
    detail: str = ""


class EvaluationMetric(ABC):
    @abstractmethod
    def score(self, generated: str, reference: str) -> MetricResult:
        raise NotImplementedError


class BleuMetric(EvaluationMetric):
    def score(self, generated: str, reference: str) -> MetricResult:
        from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

        reference_tokens = reference.split()
        generated_tokens = generated.split()
        value = sentence_bleu(
            [reference_tokens],
            generated_tokens,
            smoothing_function=SmoothingFunction().method1,
        )
        return MetricResult(name="bleu", score=round(value, 4))


class RougeLMetric(EvaluationMetric):
    def score(self, generated: str, reference: str) -> MetricResult:
        from rouge_score import rouge_scorer

        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        result = scorer.score(reference, generated)
        value = result["rougeL"].fmeasure
        return MetricResult(name="rouge_l", score=round(value, 4))


class BertScoreMetric(EvaluationMetric):
    def score(self, generated: str, reference: str) -> MetricResult:
        from bert_score import score as bert_score_fn

        _, _, f1 = bert_score_fn(
            [generated], [reference], lang="en", model_type="distilbert-base-uncased", verbose=False
        )
        value = f1.item()
        return MetricResult(name="bert_score", score=round(value, 4))


class WeightedEvaluator:
    def __init__(self, metrics: list[EvaluationMetric], weights: dict[str, float]) -> None:
        self.metrics = metrics
        self.weights = weights

    def evaluate(self, generated: str, reference: str) -> dict:
        results = [metric.score(generated, reference) for metric in self.metrics]
        overall_score = sum(
            result.score * self.weights.get(result.name, 0.0) for result in results
        )
        return {
            "components": {result.name: result.score for result in results},
            "overall_score": round(overall_score, 4),
        }