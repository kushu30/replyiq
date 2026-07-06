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

CASES = [
    ("GOOD — specific, resolves issue, no fabrication", GOOD_REPLY),
    ("BAD — generic, no specifics, no real resolution", BAD_REPLY),
    ("HALLUCINATED — confident but fabricated cause", HALLUCINATED_REPLY),
]


def run_calibration() -> None:
    judge = LlmJudgeMetric()

    print(f"{'Case':52} {'Overall':>8} {'Groundedness':>13}  Weakness identified")
    print("-" * 110)

    results = []
    for label, reply in CASES:
        result = judge.score(generated=reply, reference="", customer_email=CUSTOMER_EMAIL)
        detail = json.loads(result.detail)
        groundedness = detail.get("groundedness", "?")
        weakness = detail.get("weakness", "")
        print(f"{label:52} {result.score:>8} {groundedness:>13}  {weakness}")
        results.append((label, result.score, groundedness))

    print("\nDiscrimination check:")
    good_score = results[0][1]
    bad_score = results[1][1]
    hallucinated_groundedness = results[2][2]

    if good_score > bad_score:
        print(f"  PASS: good reply ({good_score}) scored higher than bad reply ({bad_score})")
    else:
        print(f"  FAIL: good reply ({good_score}) did NOT score higher than bad reply ({bad_score}) — judge is not discriminating")

    if isinstance(hallucinated_groundedness, (int, float)) and hallucinated_groundedness <= 3:
        print(f"  PASS: hallucinated reply's groundedness ({hallucinated_groundedness}/10) correctly flagged as low")
    else:
        print(f"  FAIL: hallucinated reply's groundedness ({hallucinated_groundedness}/10) was not flagged as low — judge missed the fabrication")


if __name__ == "__main__":
    run_calibration()