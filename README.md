# ReplyIQ — AI Email Suggested-Response System

Given an incoming customer support email, ReplyIQ retrieves similar past
emails, generates a suggested reply with an LLM, and scores the reply against
a reference-free evaluation engine — one that also evaluates retrieval quality
independently, since a bad reply can come from bad context, not just bad
generation.

**Live demo:** _add your Streamlit URL here_

## Demo
![Screenshot](<Screenshot 2026-07-07 at 04.54.05.png>)

## Architecture

```
Incoming Email
      |
      v
TF-IDF Retriever ── top-3 similar past emails
      |
      +── Retrieval Evaluator (independent of generation)
      |     domain-match precision · LLM relevance (0-10) · confidence score
      |
      v
Confidence check ── low? ──> Fallback: safe, uncertainty-admitting reply
      | high
      v
Few-shot Prompt ── Llama 3.3 70B (Groq) ── suggested reply
      |
      v
Primary Evaluation (reference-free)
  LLM Judge, 6-dimension rubric — correctness, helpfulness, completeness,
  professionalism, tone, faithfulness — with a gate that caps the score
  when fabrication is detected
      |
      v
[optional] Offline validation vs. historical reply — BLEU + ROUGE-L
      |
      v
output/report.csv
```

## Dataset

`dataset/emails.json` — 20 hand-written email/reply pairs across 10 support
domains (refunds, billing, login, password reset, cancellations, shipping,
account deletion, bugs, feature requests, payment failures).

This is synthetic by design — real tickets are private, and public corpora
don't match support tone. **The dataset size is intentional: this project
demonstrates architecture, not statistical learning.** 20 examples proves
retrieval, generation, and evaluation work correctly together; swapping in a
real 10,000-row export changes nothing else in the pipeline.

## Retrieval

TF-IDF + cosine similarity, top-3 matches. Chosen over embeddings because at
this scale it's interpretable and dependency-free — embeddings become worth it
once vocabulary mismatch (not lexical overlap) is the actual bottleneck.

## Generation

Llama 3.3 70B via Groq, prompted with retrieved examples as tone/structure
references. Two rules were added after real failures during development:

- **Never invent unconfirmed facts.** An early version confidently claimed a
  package was "misrouted" — a specific, fabricated cause. Now explicitly
  forbidden.
- **Prefer honest uncertainty over confident guessing.** "I don't have enough
  information yet, but I'm looking into it" beats a specific wrong guess.

**Low-confidence fallback:** if retrieval similarity is too weak to trust as
grounding, generation switches to a conservative prompt that drops the
unreliable examples and asks for a safe, uncertainty-admitting reply instead.

## Evaluation — the core of this project

**Primary evaluation is reference-free.** A live incoming email has no
historical "correct answer" to compare against — that's the whole premise of
generating one. The LLM judge scores the reply directly against the customer
email, no reference needed. BLEU/ROUGE-L are offline validation only,
reported separately, and never part of the primary score.

**The judge rubric has six dimensions**, not the usual five: correctness,
helpfulness, completeness, professionalism, tone, and **faithfulness** —
whether every claim is actually grounded in what the customer said, not just
plausible-sounding. A reply can be helpful and well-toned while inventing an
unconfirmed cause; faithfulness catches that specifically.

**Faithfulness gate:** a plain average let one fabricated reply (faithfulness
2/10) outscore an honest-but-generic one (faithfulness 9/10), since five good
scores diluted one bad one. Fixed: faithfulness ≤ 3 now caps the overall score
at 0.4, regardless of other dimensions. Fabrication is a severity issue, not a
quality ding.

**Judge calibration** (`evaluation/calibration_check.py`) proves this rather
than assuming it — runs the judge against a good, bad, and hallucinated reply
to the same email and checks the ranking comes out right:

```
Case                Overall  Faithfulness
GOOD                 0.8167           6
BAD (generic)         0.6833           9
HALLUCINATED          0.4              2
SUBTLE (plausible)    0.4              2

PASS: good > bad
PASS: hallucinated faithfulness correctly flagged low
PASS: fabricated reply does not outrank the honest generic one
```

**Stage-wise retrieval evaluation.** Scoring only the final reply hides
whether a bad outcome came from weak retrieval or weak generation. Two signals
run independently of the generated reply: domain-match precision
(deterministic) and an LLM relevance judgment (0-10, on a smaller model to
spread API load across a separate Groq quota). In a full run, one email's
relevance judgment was a stark 0/10 outlier while its reply quality was
unaffected — more likely a noisy small-model judgment than a real retrieval
failure, a real trade-off worth knowing about.

**Known limitations:**
- Judge and generator share a model family (Groq/Llama) — shared blind spots.
- No human-rated set yet exists to validate judge scores against real judgment.
- Faithfulness is text-only — can't verify claims like "I checked with the
  courier" against any real system.

**What this doesn't measure: business outcome.** Every metric here is a
quality proxy, not an outcome measure. A production system needs first-response
resolution rate, agent acceptance/edit distance, and CSAT — this is listed as
a gap, not solved here, since it needs production traffic this system can't
generate.

## Test Suite

33 tests across retrieval, metrics, judge (including the faithfulness gate and
its boundary condition), retrieval evaluator, generator prompts, and rate
limiting. All Groq calls are mocked — no API credits needed to run tests.

```bash
export GROQ_API_KEY=your_key_here
PYTHONPATH=. pytest tests/ -v
```

## Trade-offs

- Started on Gemini, hit a deprecated model then two free-tier quota walls.
  Migrated to Groq (70B for generation/judging, 8B for retrieval judging, on
  separate quotas).
- No fine-tuning — 20 examples would overfit, not generalize.
- BERTScore was dropped from deployment (its `torch` dependency exceeded
  Streamlit Cloud's free-tier build memory); offline validation now runs on
  BLEU + ROUGE-L only, which never affected the primary score anyway.

## Roadmap

1. Real ticket data in place of the synthetic dataset — no architecture change
2. Gmail integration (OAuth, draft creation, human approval before send)
3. Feedback loop — log agent accept/edit/reject to validate judge scores and
   eventually enable real fine-tuning
4. Embeddings retrieval once TF-IDF's vocabulary limit becomes the bottleneck
5. Cross-model judging to reduce shared blind spots
6. Business-outcome tracking (resolution rate, CSAT, agent edit distance)

## AI Tools Used

Built with Claude as a pair-programming partner: architecture design, writing
and debugging every module, migrating providers after hitting quota limits,
and — most substantively — catching real bugs (a generation hallucination,
judge leniency bias, a score-dilution bug, the reference-dependency gap) by
reading actual output rather than trusting metrics alone.

## Setup

```bash
git clone <this-repo-url>
cd replyiq
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# add your GROQ_API_KEY (free at https://console.groq.com/keys)

python main.py                                    # full pipeline
PYTHONPATH=. pytest tests/ -v                      # test suite
PYTHONPATH=. python evaluation/calibration_check.py  # judge calibration
streamlit run app.py                               # live demo
```

## Example Output

```
[1] refunds -> retrieval_relevance=7/10 primary_score=0.8667 (offline=0.9908)
[6] login_issues -> retrieval_relevance=0/10 primary_score=0.7833 (offline=0.8944)
[20] payment_failures -> retrieval_relevance=8/10 primary_score=0.7833 (offline=0.8738)

Report written to output/report.csv
Overall system score (primary, reference-free): 0.8142
```