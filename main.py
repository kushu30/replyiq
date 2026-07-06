import csv
import json
from pathlib import Path

from evaluation.judge import LlmJudgeMetric
from evaluation.metrics import BertScoreMetric, BleuMetric, RougeLMetric, WeightedEvaluator
from evaluation.retrieval_evaluator import RetrievalEvaluator
from generator.generator import ReplyGenerator
from generator.retriever import EmailRetriever

DATASET_PATH = "dataset/emails.json"
OUTPUT_PATH = "output/report.csv"
TOP_K_RETRIEVAL = 3

METRIC_WEIGHTS = {
    "bleu": 0.15,
    "rouge_l": 0.15,
    "bert_score": 0.30,
    "llm_judge": 0.40,
}


def load_test_emails(dataset_path: str) -> list[dict]:
    with open(dataset_path, "r", encoding="utf-8") as file:
        return json.load(file)


def run_pipeline(limit: int | None = None) -> None:
    retriever = EmailRetriever(DATASET_PATH)
    generator = ReplyGenerator()
    lexical_evaluator = WeightedEvaluator(
        metrics=[BleuMetric(), RougeLMetric(), BertScoreMetric()],
        weights=METRIC_WEIGHTS,
    )
    judge = LlmJudgeMetric()
    retrieval_evaluator = RetrievalEvaluator()

    test_emails = load_test_emails(DATASET_PATH)
    if limit:
        test_emails = test_emails[:limit]

    report_rows = []
    system_scores = []

    for email in test_emails:
        customer_email = email["customer_email"]
        reference_reply = email["support_reply"]

        similar_examples = retriever.retrieve_similar(customer_email, top_k=TOP_K_RETRIEVAL)
        domain_precision = retrieval_evaluator.domain_precision(email["domain"], similar_examples)
        retrieval_judgment = retrieval_evaluator.judge_relevance(customer_email, similar_examples)

        generated_reply = generator.generate_reply(customer_email, similar_examples)

        lexical_result = lexical_evaluator.evaluate(generated_reply, reference_reply)
        judge_result = judge.score(generated_reply, reference_reply, customer_email)

        overall_score = round(
            lexical_result["overall_score"] + judge_result.score * METRIC_WEIGHTS["llm_judge"],
            4,
        )

        report_rows.append(
            {
                "id": email["id"],
                "domain": email["domain"],
                "customer_email": customer_email,
                "generated_reply": generated_reply,
                "reference_reply": reference_reply,
                "retrieval_domain_precision": domain_precision,
                "retrieval_relevance_score": retrieval_judgment.get("relevance_score", 0),
                "retrieval_reasoning": retrieval_judgment.get("reasoning", ""),
                "bleu": lexical_result["components"]["bleu"],
                "rouge_l": lexical_result["components"]["rouge_l"],
                "bert_score": lexical_result["components"]["bert_score"],
                "llm_judge": judge_result.score,
                "judge_reasoning": judge_result.detail,
                "overall_score": overall_score,
            }
        )
        system_scores.append(overall_score)

        print(
            f"[{email['id']}] {email['domain']} -> "
            f"retrieval_precision={domain_precision} "
            f"retrieval_relevance={retrieval_judgment.get('relevance_score', 0)}/10 "
            f"overall_score={overall_score}"
        )

    write_report(report_rows, system_scores)


def write_report(rows: list[dict], system_scores: list[float]) -> None:
    Path("output").mkdir(exist_ok=True)
    fieldnames = list(rows[0].keys())

    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    average_score = round(sum(system_scores) / len(system_scores), 4)
    print(f"\nReport written to {OUTPUT_PATH}")
    print(f"Overall system score: {average_score}")


if __name__ == "__main__":
    run_pipeline()