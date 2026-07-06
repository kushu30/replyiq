import json

from evaluation.judge import LlmJudgeMetric

CUSTOMER_EMAIL = (
    "My order #48213 was supposed to arrive 4 days ago and the tracking page "
    "hasn't updated at all. I have no idea where my package is and I'm getting worried."
)

GOOD_REPLY = (
    "I'm sorry for the lack of updates on order #48213, I understand that's worrying. "
    "I checked directly with our courier and your package is confirmed in transit, "
    "currently at a regional hub, with delivery expected within 2 business days. "
    "I'll personally monitor this and email you the moment it's out for delivery, and "
    "if it hasn't arrived by then I'll escalate for a replacement or refund immediately."
)

BAD_REPLY = (
    "Thank you for reaching out. We apologize for any inconvenience. Our team is "
    "looking into this and will get back to you soon. Thank you for your patience."
)

HALLUCINATED_REPLY = (
    "I'm sorry for the delay. I've confirmed with our warehouse that your package "
    "was destroyed in a fire at our regional distribution center last week, so we've "
    "shipped a full replacement which will arrive within 24 hours."
)

SUBTLE_UNFAITHFUL_EMAIL = "My package is late."
SUBTLE_UNFAITHFUL_REPLY = (
    "I'm sorry for the delay. Your package was delayed because the warehouse "
    "closed earlier than expected. It should arrive shortly."
)

CASES = [
    ("GOOD — specific, resolves issue, no fabrication", GOOD_REPLY),
    ("BAD — generic, no specifics, no real resolution", BAD_REPLY),
    ("HALLUCINATED — confident but fabricated cause", HALLUCINATED_REPLY),
]


def run_calibration() -> None:
    judge = LlmJudgeMetric()

    print(f"{'Case':52} {'Overall':>8} {'Faithfulness':>13}  Weakness identified")
    print("-" * 110)

    results = []
    for label, reply in CASES:
        result = judge.score(generated=reply, reference="", customer_email=CUSTOMER_EMAIL)
        detail = json.loads(result.detail)
        faithfulness = detail.get("faithfulness", "?")
        weakness = detail.get("weakness", "")
        print(f"{label:52} {result.score:>8} {faithfulness:>13}  {weakness}")
        results.append((label, result.score, faithfulness))

    subtle_label = "SUBTLE — plausible cause, still unconfirmed by customer"
    subtle_result = judge.score(generated=SUBTLE_UNFAITHFUL_REPLY, reference="", customer_email=SUBTLE_UNFAITHFUL_EMAIL)
    subtle_detail = json.loads(subtle_result.detail)
    subtle_faithfulness = subtle_detail.get("faithfulness", "?")
    print(f"{subtle_label:52} {subtle_result.score:>8} {subtle_faithfulness:>13}  {subtle_detail.get('weakness', '')}")

    print("\nDiscrimination check:")
    good_score = results[0][1]
    bad_score = results[1][1]
    hallucinated_faithfulness = results[2][2]

    if good_score > bad_score:
        print(f"  PASS: good reply ({good_score}) scored higher than bad reply ({bad_score})")
    else:
        print(f"  FAIL: good reply ({good_score}) did NOT score higher than bad reply ({bad_score}) — judge is not discriminating")

    if isinstance(hallucinated_faithfulness, (int, float)) and hallucinated_faithfulness <= 3:
        print(f"  PASS: hallucinated reply's faithfulness ({hallucinated_faithfulness}/10) correctly flagged as low")
    else:
        print(f"  FAIL: hallucinated reply's faithfulness ({hallucinated_faithfulness}/10) was not flagged as low — judge missed the fabrication")

    hallucinated_overall = results[2][1]
    if hallucinated_overall <= bad_score:
        print(f"  PASS: fabricated reply ({hallucinated_overall}) does NOT outrank the merely-generic reply ({bad_score}) — faithfulness gate is working")
    else:
        print(f"  FAIL: fabricated reply ({hallucinated_overall}) outranks the merely-generic reply ({bad_score}) — a confident lie is beating an honest mediocre reply, faithfulness gate needs tuning")


if __name__ == "__main__":
    run_calibration()