import streamlit as st

from evaluation.judge import LlmJudgeMetric
from evaluation.metrics import BertScoreMetric, BleuMetric, RougeLMetric, WeightedEvaluator
from evaluation.retrieval_evaluator import RetrievalEvaluator
from generator.generator import ReplyGenerator
from generator.retriever import EmailRetriever

DATASET_PATH = "dataset/emails.json"
OFFLINE_METRIC_WEIGHTS = {"bleu": 0.15, "rouge_l": 0.15, "bert_score": 0.70}
CONFIDENCE_THRESHOLD = 0.15

st.set_page_config(page_title="ReplyIQ", layout="wide")


@st.cache_resource
def load_pipeline():
    retriever = EmailRetriever(DATASET_PATH)
    generator = ReplyGenerator()
    offline_evaluator = WeightedEvaluator(
        metrics=[BleuMetric(), RougeLMetric(), BertScoreMetric()],
        weights=OFFLINE_METRIC_WEIGHTS,
    )
    judge = LlmJudgeMetric()
    retrieval_evaluator = RetrievalEvaluator()
    return retriever, generator, offline_evaluator, judge, retrieval_evaluator


st.title("ReplyIQ — Live Suggested Reply Demo")
st.caption("Paste a real customer email below. Retrieval, generation, and scoring all run live.")

retriever, generator, offline_evaluator, judge, retrieval_evaluator = load_pipeline()

customer_email = st.text_area(
    "Incoming customer email",
    height=140,
    placeholder="e.g. My order arrived broken and I want a refund, not a replacement.",
)

reference_reply = st.text_area(
    "Optional: historical reference reply, for offline validation only (leave blank to skip)",
    height=100,
)

if st.button("Generate suggested reply", type="primary") and customer_email.strip():
    with st.spinner("Retrieving similar past emails..."):
        similar_examples = retriever.retrieve_similar(customer_email, top_k=3)
        confidence = retrieval_evaluator.confidence_score(similar_examples)
        low_confidence = confidence < CONFIDENCE_THRESHOLD

    st.metric("Retrieval confidence", f"{confidence:.0%}")
    if low_confidence:
        st.warning(
            "Low confidence — no sufficiently similar historical emails found. "
            "Falling back to a generic, safe reply rather than inventing specifics."
        )

    with st.expander("Retrieved similar emails (grounding context)"):
        for example in similar_examples:
            st.markdown(f"**Similarity: {example['similarity_score']}** — *{example['domain']}*")
            st.write(f"Customer: {example['customer_email']}")
            st.write(f"Reply: {example['support_reply']}")
            st.divider()

    with st.spinner("Generating reply..."):
        generated_reply = generator.generate_reply(customer_email, similar_examples, low_confidence=low_confidence)

    st.subheader("Suggested reply")
    st.info(generated_reply)

    with st.spinner("Scoring reply..."):
        primary_result = judge.score(generated=generated_reply, reference="", customer_email=customer_email)

        offline_result = None
        if reference_reply.strip():
            offline_result = offline_evaluator.evaluate(generated_reply, reference_reply)

    st.subheader("Evaluation")
    st.caption("Primary score is reference-free — this is what would run in production, where no ground-truth reply exists.")
    cols = st.columns(2)
    cols[0].metric("Primary score (reference-free judge)", primary_result.score)
    cols[1].metric("Confidence", f"{confidence:.0%}")

    if offline_result:
        st.caption("Offline validation only — requires a historical reference, not available in production.")
        offline_cols = st.columns(4)
        offline_cols[0].metric("BLEU", offline_result["components"]["bleu"])
        offline_cols[1].metric("ROUGE-L", offline_result["components"]["rouge_l"])
        offline_cols[2].metric("BERTScore", offline_result["components"]["bert_score"])
        offline_cols[3].metric("Offline overall", offline_result["overall_score"])

    with st.expander("Judge reasoning"):
        st.json(primary_result.detail)