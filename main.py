import csv
import json
from pathlib import Path

from evaluation.judge import LlmJudgeMetric
from evaluation.metrics import BleuMetric, RougeLMetric, WeightedEvaluator
from evaluation.retrieval_evaluator import RetrievalEvaluator
from generator.generator import ReplyGenerator
from generator.retriever import EmailRetriever

DATASET_PATH = "dataset/emails.json"
OUTPUT_PATH = "output/report.csv"
TOP_K_RETRIEVAL = 3
CONFIDENCE_THRESHOLD = 0.15

OFFLINE_METRIC_WEIGHTS = {"bleu": 0.5, "rouge_l": 0.5}


def load_test_emails(dataset_path: str) -> list[dict]:
    with open(dataset_path, "r", encoding="utf-8") as file:
        return json.load(file)


def run_pipeline(limit: int | None = None) -> None:
    retriever = EmailRetriever(DATASET_PATH)
    generator = ReplyGenerator()
    judge = LlmJudgeMetric()
    retrieval_evaluator = RetrievalEvaluator()
    offline_evaluator = WeightedEvaluator(
        metrics=[BleuMetric(), RougeLMetric()],
        weights=OFFLINE_METRIC_WEIGHTS,
    )

    test_emails = load_test_emails(DATASET_PATH)
    if limit:
        test_emails = test_emails[:limit]

    report_rows = []
    primary_scores = []

    for email in test_emails:
        customer_email = email["customer_email"]
        historical_reply = email["support_reply"]

        similar_examples = retriever.retrieve_similar(customer_email, top_k=TOP_K_RETRIEVAL)
        domain_precision = retrieval_evaluator.domain_precision(email["domain"], similar_examples)
        retrieval_judgment = retrieval_evaluator.judge_relevance(customer_email, similar_examples)
        confidence = retrieval_evaluator.confidence_score(similar_examples)
        low_confidence = confidence < CONFIDENCE_THRESHOLD

        if low_confidence:
            print(f"  ⚠ Low confidence ({confidence}) — no sufficiently similar examples found, falling back to a generic safe reply")

        generated_reply = generator.generate_reply(customer_email, similar_examples, low_confidence=low_confidence)

        # PRIMARY EVALUATION: reference-free. This is the score that would run in
        # production, where no historical reply exists to compare against — the
        # entire point of the system is to generate a reply where none exists yet.
        primary_result = judge.score(generated=generated_reply, reference="", customer_email=customer_email)

        # OFFLINE VALIDATION ONLY: lexical/semantic comparison against the historical
        # reply. This is only possible here because this dataset happens to have
        # ground truth. It is a development-time sanity check, not part of the
        # production scoring path, and is reported separately for that reason.
        offline_result = offline_evaluator.evaluate(generated_reply, historical_reply)

        report_rows.append(
            {
                "id": email["id"],
                "domain": email["domain"],
                "customer_email": customer_email,
                "generated_reply": generated_reply,
                "historical_reference_reply": historical_reply,
                "retrieval_domain_precision": domain_precision,
                "retrieval_relevance_score": retrieval_judgment.get("relevance_score", 0),
                "retrieval_reasoning": retrieval_judgment.get("reasoning", ""),
                "confidence_score": confidence,
                "low_confidence_fallback_used": low_confidence,
                "primary_score": primary_result.score,
                "judge_reasoning": primary_result.detail,
                "offline_bleu": offline_result["components"]["bleu"],
                "offline_rouge_l": offline_result["components"]["rouge_l"],
                "offline_validation_score": offline_result["overall_score"],
            }
        )
        primary_scores.append(primary_result.score)

        print(
            f"[{email['id']}] {email['domain']} -> "
            f"retrieval_relevance={retrieval_judgment.get('relevance_score', 0)}/10 "
            f"primary_score={primary_result.score} "
            f"(offline_validation={offline_result['overall_score']})"
        )

    write_report(report_rows, primary_scores)


def write_report(rows: list[dict], primary_scores: list[float]) -> None:
    Path("output").mkdir(exist_ok=True)
    fieldnames = list(rows[0].keys())

    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    average_primary_score = round(sum(primary_scores) / len(primary_scores), 4)
    print(f"\nReport written to {OUTPUT_PATH}")
    print(f"Overall system score (primary, reference-free): {average_primary_score}")


if __name__ == "__main__":
    run_pipeline()