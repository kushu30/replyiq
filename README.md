# ReplyIQ — AI Email Suggested-Response System

An end-to-end system that takes an incoming customer support email, retrieves
similar past emails, generates a suggested reply with an LLM, and scores both the
retrieval and the reply using a stage-aware, multi-signal evaluation engine.

## Problem Statement

Given a new customer email, suggest a reply an agent could send with minimal
editing — grounded in how the team has actually replied before, not generic LLM
output. Just as important: know *why* a suggestion is good or bad. A single
blended quality score isn't enough on its own — you need to be able to tell
whether a bad reply is a retrieval problem or a generation problem, since they get
fixed in completely different ways.

## Architecture

```
Incoming Email
      |
      v
TF-IDF Retriever  --> top-3 similar past emails (customer + reply pairs)
      |
      +---------------------------------------------+
      |                                              |
      v                                              v
Prompt Builder                          Retrieval Evaluator
      |                                    |-- domain-match precision (deterministic)
      v                                    |-- LLM relevance judgment (0-10)
Llama 3.3 70B (Groq) --> suggested reply           |
      |                                              |
      v                                              |
Evaluation Engine  <----------------------------------
  |-- BLEU        (lexical n-gram overlap)
  |-- ROUGE-L     (longest common subsequence overlap)
  |-- BERTScore    (semantic embedding similarity)
  |-- LLM Judge   (6-dimension rubric incl. faithfulness)
      |
      v
Weighted Overall Score --> output/report.csv (per-email + system average)
```

The retrieval stage is evaluated independently of the generation stage — this is
the key design decision in this system (see "Stage-Wise Evaluation" below).

## Dataset

`dataset/emails.json` — 20 hand-authored customer email + support reply pairs
across 10 domains (refunds, billing, login issues, password reset, subscription
cancellation, shipping delays, account deletion, technical bugs, feature requests,
payment failures), 2 examples per domain.

**Why synthetic and hand-authored, not scraped:** real support tickets are
private and identifiable; public email corpora don't reflect support-specific
tone or structure. I wrote realistic examples aiming for real variance — different
customer tones (frustrated, neutral, polite), different reply lengths, and a
consistent professional-support voice.

**Honest limitation:** 20 examples proves the pipeline and retrieval logic work
correctly; it does not prove statistical robustness. The architecture is
unchanged if this were swapped for a real 10,000-row ticket export — only the
JSON loader would need to point elsewhere.

## Retrieval Strategy

TF-IDF + cosine similarity over customer email text, retrieving the top-3 most
similar past emails as few-shot context for generation.

**Why TF-IDF over embeddings:** at this dataset size, TF-IDF gives interpretable,
debuggable similarity (you can see exactly which words drove a match) with zero
external dependencies. Embeddings would help once vocabulary mismatch becomes the
bottleneck (e.g. matching "cancel my plan" to "how do I stop my subscription"
without shared words) — that's the first thing to swap in at larger scale.

## Prompt Engineering

The generation prompt instructs the model to use retrieved examples as tone and
structure references without copying them verbatim, output only the reply text
with no preamble, and — critically — **never invent specific unconfirmed facts**.

This last constraint was added after a real failure: an early version of the
system generated a reply confidently claiming a customer's package was
"misrouted" — a specific cause the customer never mentioned and nothing in the
dataset confirmed. This is a textbook hallucination: it sounds helpful and
specific, and it's fabricated. The fix was adding an explicit instruction to hedge
("say you're looking into it") rather than assert unconfirmed causes, dates, or
order details.

## Evaluation Methodology (the core of this challenge)

### Why exact match / lexical overlap alone is insufficient

Two replies can use completely different wording and both be excellent — or
share most of the same words and still miss the customer's actual issue. Good
evaluation needs to capture both *appropriate language* and *actual problem
resolution*.

### Stage-wise evaluation

The first version of this system only scored the final generated reply. That
conflates two independent failure modes: **the retriever fed the LLM weak
context**, or **the LLM wrote a poor reply despite good context**. A single
blended score can't distinguish these, which makes the system undebuggable —
you'd blame the LLM every time, even when the real fix is in retrieval.

Two retrieval-stage signals were added, both computed independently of the final
reply:

- **Domain-match precision** (deterministic, free): what fraction of the top-3
  retrieved examples share the query's actual domain. Catches obvious retrieval
  misses instantly, no API call needed.
- **LLM relevance judgment** (0-10): an LLM is asked *only* whether the retrieved
  examples are useful context for the new email — it never sees the generated
  reply, so this can't be contaminated by generation quality.

**What this actually caught, from a real run:** one email had a domain-match
precision of only 0.333 (1 of 3 retrieved examples off-domain) and the lowest LLM
relevance score in the batch (4/10) — and it also had one of the lowest overall
reply scores. That's a traceable causal chain: weak retrieval → weak grounding →
weak reply, confirmed by two independent signals rather than assumed. Another
email had strong retrieval (8/10 relevance) and the highest overall score in the
batch. This is exactly the diagnostic capability a single end-to-end score
cannot provide.

### Primary evaluation is reference-free — this is a deliberate architectural choice

In production, reference replies do not exist. That's literally why the system
is generating a reply in the first place — there is no historical "correct
answer" for a brand-new incoming email. Therefore our primary evaluation is
reference-free: the LLM judge scores the generated reply directly against the
customer email and general support standards, with no comparison to a reference.

BLEU, ROUGE-L, and BERTScore are offline validation tools only. They require a
reference reply, which is only available here because this dataset happens to
include historical ground truth. They are reported as `offline_*` columns for
development-time sanity checking — confirming the generator stays reasonably
close to how the team has actually replied before — but they are never part of
the headline `primary_score`, and could not be computed at all on a genuinely
new email in production.

### Offline validation weighting — why these numbers, not invented ones

The offline validation blend combines BLEU, ROUGE-L, and BERTScore with a
justified structure rather than arbitrary numbers:

- **BLEU measures lexical similarity** — lowest importance, since a correctly
  paraphrased reply can score low here despite being excellent.
- **ROUGE-L captures structural overlap** — still lexical, still low importance
  for the same reason.
- **BERTScore captures semantic equivalence** — higher importance, because it
  tolerates paraphrasing while still penalizing replies that are actually wrong.

Grouped this way, BLEU and ROUGE-L together form a **30% lexical** weight, and
BERTScore alone forms a **70% semantic** weight. The reasoning is direct:
support replies should be judged primarily on whether they convey the right
meaning, not on whether they use the same words as a historical reply. This
30/70 split applies only to offline validation — it never touches the primary,
reference-free score, which has no weights to justify because it's a single
signal (the judge), not a blend.

### Judge calibration — proving discrimination, not just claiming it

An LLM-as-judge is only trustworthy if it can be shown to actually discriminate
between good, bad, and misleading replies — otherwise it's indistinguishable
from a judge that just returns a high number regardless of input.
`evaluation/calibration_check.py` runs the judge, reference-free, against three
hand-crafted replies to the same customer email:

- a **good** reply: specific, references the actual order, gives a concrete
  resolution and timeline, no fabrication
- a **bad** reply: generic, templated, no specifics, no real resolution
- a **hallucinated** reply: reads as confident and helpful, but invents an
  unconfirmed cause (a fabricated warehouse fire)

Run it with:
```bash
python evaluation/calibration_check.py
```

**Results from an actual run:**

```
<PASTE_CALIBRATION_OUTPUT_HERE>
```

The script explicitly checks two things: that the good reply scores higher than
the bad reply (basic discrimination), and that the hallucinated reply's
faithfulness score is flagged low despite being fluent and well-structured
(catching fabrication specifically, not just general "quality"). Both checks
passing is what justifies trusting the judge's scores elsewhere in this system —
without this check, "we use an LLM as judge" is just an assertion.

**The judge rubric has six dimensions**, not five:
correctness, helpfulness, completeness, professionalism, tone, and
**faithfulness to customer-provided information** — whether every claim in the
reply is actually grounded in what the customer said, as opposed to being
merely plausible-sounding. This is deliberately named "faithfulness" rather
than the more common "groundedness," because the relevant source of truth here
is specifically what the customer told us, not external world knowledge.

The distinction matters because faithfulness failures don't look like failures
on any other dimension. Consider: a customer writes only "my package is late."
A reply says "your package was delayed because the warehouse closed." That
reply is helpful (yes), correct (unknown — the cause was never confirmed),
well-toned (excellent), and yet not grounded in anything the customer actually
said. A rubric without a dedicated faithfulness dimension would likely score
that reply highly across the board, because everything about it *reads* well.
We separately evaluate faithfulness to customer-provided information for
exactly this reason — a reply that invents a specific unconfirmed cause must
score low on faithfulness (0-3) regardless of how good it looks on every other
axis. Splitting "sounds good" from "is actually grounded" into separate
dimensions is deliberate — a single quality score would hide exactly this kind
of failure.

### Validating the judge isn't just agreeable

The first version of the judge prompt gave a uniform 10/10/10/10/10 on a
genuinely mediocre, generic reply. This is a well-documented LLM-as-judge failure
mode — leniency bias, where the judge defaults to agreeable, undifferentiated
scores instead of actually discriminating. It was caught by manually reading the
judge's own reasoning field and noticing it never identified a single weakness,
on any email, ever.

The fix: the prompt now explicitly instructs the judge to act as a skeptical
reviewer, reserve 9-10 for replies with no meaningful flaw at all, and the JSON
schema requires a `weakness` field naming the single most significant flaw in
every reply — the judge cannot skip identifying a problem.

**Effect of these fixes on the score:** the system's overall average dropped from
an earlier ~0.90 to ~0.67 after fixing judge leniency and adding faithfulness.
This is a improvement in scoring honesty, not a regression in system quality —
a lower, more discriminating score is more trustworthy than an inflated one.

### Reference-free judging

Historical emails have a ground-truth reply to compare against; a brand-new
incoming email in production does not. The judge was updated to work without a
reference reply — the correctness/helpfulness/completeness/professionalism/
tone/faithfulness rubric can be assessed directly against the customer email and
general support standards, with no comparison needed. Only BLEU/ROUGE-L/
BERTScore genuinely require a reference and are skipped when none is available.

### Known limitations

- The LLM judge and the generator currently use models from the same provider
  (Groq/Llama), which risks shared blind spots. A more rigorous setup would use a
  different model family as judge.
- No held-out human-rated set exists yet to formally correlate judge scores
  against human judgment — this is the single most important next validation
  step (see Roadmap).
- BERTScore uses `distilbert-base-uncased` rather than the default
  `roberta-large`, trading some accuracy for a much smaller, faster download.

## Reporting

`output/report.csv` contains, per email: the customer email, generated reply,
reference reply, retrieval domain precision, retrieval relevance score and
reasoning, all four reply-level component scores, the judge's structured
reasoning (including the named weakness), and the weighted overall score. The
console output prints retrieval and overall scores per email as they run, plus a
system-wide average at the end.

## Test Suite

`tests/` contains 24 unit tests covering:
- Retrieval correctness and ranking (`test_retriever.py`)
- BLEU/ROUGE-L scoring behavior and the weighted evaluator's aggregation logic
  (`test_metrics.py`)
- Judge response parsing (valid JSON, malformed JSON, markdown-fenced JSON),
  score averaging, and reference-free scoring, using mocked Groq responses so no
  API calls or credits are needed to run tests (`test_judge.py`)
- Retrieval evaluator domain-precision math and mocked relevance judging
  (`test_retrieval_evaluator.py`)
- The rate-limit retry decorator's backoff and eventual-failure behavior
  (`test_rate_limit.py`)

BERTScore is intentionally not covered by a unit test — it requires a ~250MB
model download on first run, which is too slow and flaky for a fast test suite;
it's validated via full pipeline runs instead.

Run with:
```bash
export GROQ_API_KEY=your_key_here
PYTHONPATH=. pytest tests/ -v
```

## Trade-offs

- **Model migration mid-build:** started on Gemini, hit a deprecated model name,
  then hit free-tier daily quota limits twice (5 req/min, then 20 req/day).
  Migrated generation and both judges to Groq (Llama 3.3 70B for generation and
  reply judging, Llama 3.1 8B for retrieval judging) partly for more generous
  free-tier limits, and partly to deliberately spread API calls across two
  separate model quotas rather than one.
- **No fine-tuning:** with only 20 examples, fine-tuning would overfit rather
  than generalize. Few-shot retrieval-grounded prompting is the correct choice at
  this data scale, not just the faster one.
- **Retry-with-backoff** (`generator/rate_limit.py`) wraps every LLM call — a
  real production resilience pattern, not just a workaround for this build.

## Roadmap to Production

**1. Real data.** Replace `dataset/emails.json` with a loader against a real
historical ticket export (Gmail/Zendesk/Hiver). The retriever, generator, and
evaluator are all unchanged — only the data source changes.

**2. Gmail integration.** Read incoming mail via the Gmail API (OAuth2,
`readonly` for ingestion, `compose` for draft creation), run each new email
through the existing pipeline, and write the suggested reply back as a Gmail
draft — a human still approves before send.

**3. Feedback loop.** Log `(generated_reply, final_sent_reply, accepted: bool)`
every time an agent edits or accepts a suggestion. This becomes a real human
quality signal, usable to (a) check whether the LLM judge score actually
correlates with acceptance rate, recalibrating metric weights if not, and (b)
eventually provide real training data for fine-tuning once there's enough volume.

**4. Embeddings retrieval.** Swap in once the dataset is large enough that
TF-IDF's vocabulary-overlap limitation becomes the actual bottleneck.

**5. Cross-model judging.** Use a different model family as judge (not the same
one generating replies) to reduce shared blind spots.

## AI Tools Used

Built with Claude as a pair-programming and architecture partner throughout:
designing the Strategy-pattern evaluation engine, writing and debugging the
retriever/generator/judge/retrieval-evaluator modules, diagnosing a Gemini model
deprecation and two separate free-tier quota walls, migrating the LLM provider to
Groq, and — most substantively — catching and fixing two real quality bugs (judge
leniency bias and a generation hallucination) by manually reading actual output
rather than trusting metric scores alone. All code was reviewed and adjusted by
me.

## Setup Instructions

```bash
git clone <this-repo-url>
cd replyiq
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# edit .env and add your GROQ_API_KEY (free at https://console.groq.com/keys)
python main.py
```

Output is written to `output/report.csv`; the overall system score prints to the
console at the end of the run.

Run the test suite:
```bash
PYTHONPATH=. pytest tests/ -v
```

Run the interactive live demo:
```bash
streamlit run app.py
```

## Example Output

```
[1] refunds -> retrieval_precision=0.6667 retrieval_relevance=8/10 overall_score=0.9469
[2] refunds -> retrieval_precision=0.6667 retrieval_relevance=8/10 overall_score=0.6497
[3] billing -> retrieval_precision=0.3333 retrieval_relevance=8/10 overall_score=0.6201
[4] billing -> retrieval_precision=0.6667 retrieval_relevance=8/10 overall_score=0.5145
[5] login_issues -> retrieval_precision=0.3333 retrieval_relevance=4/10 overall_score=0.6065

Report written to output/report.csv
Overall system score: 0.6675
```