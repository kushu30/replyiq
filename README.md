# ReplyIQ — AI Email Suggested-Response System

An end-to-end system that takes an incoming customer support email, retrieves similar
past emails, generates a suggested reply with Gemini, and scores that reply against
a reference reply using a combination of lexical, semantic, and LLM-judged metrics.

## Problem Statement

Given a new customer email, suggest a reply that a support agent could send with
minimal editing — grounded in how the team has actually replied before, not just
generic LLM output. Just as important: quantify how good each suggestion is, in a
way that reflects real reply quality, not just surface word overlap.

## Architecture

```
Incoming Email
      |
      v
TF-IDF Retriever  --> top-3 similar past emails (customer + reply pairs)
      |
      v
Prompt Builder --> few-shot prompt grounded in retrieved examples
      |
      v
Gemini (gemini-2.5-flash-lite) --> suggested reply
      |
      v
Evaluation Engine
  |-- BLEU        (lexical n-gram overlap)
  |-- ROUGE-L     (longest common subsequence overlap)
  |-- BERTScore   (semantic embedding similarity)
  |-- LLM Judge   (rubric: correctness, helpfulness, completeness, professionalism, tone)
      |
      v
Weighted Overall Score --> output/report.csv (per-email + system average)
```

## Dataset

`dataset/emails.json` — 20 hand-authored customer email + support reply pairs across
10 domains (refunds, billing, login issues, password reset, subscription
cancellation, shipping delays, account deletion, technical bugs, feature requests,
payment failures), 2 examples per domain.

**Why synthetic and hand-authored, not scraped:** real support tickets are private
and identifiable; public corpora (e.g. generic email datasets) don't reflect the
tone or structure of B2B/B2C support replies specifically. I wrote realistic
examples instead, aiming for the variance you'd actually see — different customer
tones (frustrated, neutral, polite), different reply lengths, and replies that
follow a consistent professional-support voice. This is a limitation I'm explicit
about: 20 examples is enough to demonstrate the pipeline and retrieval logic
working correctly, not enough to claim statistical robustness. The architecture
does not change if this were swapped for a real 10,000-row support ticket export.

## Retrieval Strategy

TF-IDF + cosine similarity over customer email text, retrieving the top-3 most
similar past emails as few-shot context for generation.

**Why TF-IDF over embeddings:** at this dataset size (20 rows), TF-IDF gives
interpretable, debuggable similarity (you can inspect exactly which words drove a
match) with zero external dependencies or API calls. Embedding-based retrieval
would improve semantic matching (e.g. matching "cancel my plan" to "how do I stop
my subscription" without shared vocabulary) but is not justified until the dataset
is large enough that lexical overlap breaks down. This is the first thing I'd swap
in a production version.

## Prompt Engineering

The generation prompt explicitly instructs the model to use retrieved examples as
tone/structure references without copying them verbatim, and to output only the
reply text with no preamble — this keeps output clean for scoring and avoids the
model parroting retrieved replies instead of addressing the new email's specific
issue.

## Evaluation Methodology (the core of this challenge)

**Why exact match / lexical overlap alone is insufficient:** two replies can use
completely different wording and both be excellent — or share most of the same
words and still miss the customer's actual issue. A good evaluation has to capture
both *did it use appropriate language* and *did it actually solve the problem*.

I combine four signals:

| Metric | Captures | Weight |
|---|---|---|
| BLEU | n-gram precision vs. reference reply | 0.15 |
| ROUGE-L | longest common subsequence, phrasing structure | 0.15 |
| BERTScore | semantic similarity even with different wording | 0.30 |
| LLM Judge | correctness, helpfulness, completeness, professionalism, tone | 0.40 |

**Why these weights:** lexical metrics (BLEU/ROUGE-L) are the weakest signal for
this task — a paraphrased-but-correct reply can score low on them — so they're
weighted lowest but kept as a sanity floor (a reply scoring near-zero on both
usually means something went structurally wrong, like the model ignoring the
question). BERTScore is weighted higher because it tolerates paraphrasing while
still penalizing semantically wrong replies. The LLM Judge gets the highest weight
because it's the only metric actually reading the reply for whether it *resolves*
the customer's issue — the thing we actually care about — scored against a
5-dimension rubric (correctness, helpfulness, completeness, professionalism, tone)
rather than a single vague "quality" number, which makes the score explainable
rather than a black box.

**How I validated the metric isn't just a number:** I spot-checked cases where
lexical and semantic/judge scores diverged. For example, in row 4 of the output
report, the generated reply used different phrasing than the reference (lower
BLEU) but was rated highly by both BERTScore and the judge — reading it manually
confirmed it correctly resolved the issue, just in different words. That divergence
is the evaluation system doing its job: catching that lexical overlap would have
under-scored a genuinely good reply. I did not have time in this challenge to build
a held-out human-rating set to formally correlate judge scores against human
judgment — that's the single most important next step to trust this system at
scale (see Future Improvements).

**Known limitation:** the LLM judge is Gemini judging Gemini's own output family.
A more rigorous setup would use a different model (or human raters) as the judge
to avoid shared blind spots.

## Reporting

`output/report.csv` contains, per email: the customer email, generated reply,
reference reply, all four component scores, the judge's structured reasoning, and
a weighted overall score. The console output also prints a system-wide average
score across all evaluated emails.

## Trade-offs (given the time constraint)

- Evaluated on a subset of the dataset (8 of 20 emails) due to Gemini free-tier
  daily quota limits (20 requests/day on `gemini-2.5-flash`, hit mid-run). Switched
  to `gemini-2.5-flash-lite` for a separate quota bucket and added retry-with-backoff
  handling for rate limits. The pipeline itself is unchanged and scales to the full
  dataset — this is purely an API quota constraint, not an architecture limitation.
- No fine-tuning: few-shot retrieval-grounded prompting was the faster, more
  transparent choice for this timeframe, and doesn't require training
  infrastructure or risk overfitting to 20 examples.
- BERTScore uses `distilbert-base-uncased` instead of the default `roberta-large`
  to keep first-run download time low; a production version would use a stronger
  model.

## Future Improvements

- Build a small held-out set of human-rated replies and correlate against the LLM
  judge score to validate (or recalibrate) the weighting.
- Swap TF-IDF for embedding-based retrieval once the dataset is large enough that
  vocabulary mismatch becomes the bottleneck.
- Use a different (or multiple) LLM as judge to reduce self-evaluation bias.
- Expand the dataset and re-run on the full set once quota allows.

## AI Tools Used

Built with Claude as a pair-programming and architecture-design partner: designing
the Strategy-pattern evaluation engine, writing/debugging the retriever, generator,
and judge modules, and diagnosing Gemini API model deprecation and rate-limit
issues encountered during the build. All code was reviewed and adjusted by me.

## Setup Instructions

```bash
git clone <this-repo-url>
cd replyiq
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# edit .env and add your GEMINI_API_KEY
python main.py
```

Output is written to `output/report.csv`, and the overall system score is printed
to the console at the end of the run.

## Example Output

```
[1] refunds -> overall_score=1.0
[2] refunds -> overall_score=0.8006
[3] billing -> overall_score=0.7827
...
Report written to output/report.csv
Overall system score: 0.8953
```