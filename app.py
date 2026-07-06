import json
import os

import streamlit as st

try:
    if "GROQ_API_KEY" in st.secrets:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
except Exception:
    pass  # No secrets.toml locally — .env / environment variables are used instead

from evaluation.judge import LlmJudgeMetric
from evaluation.metrics import BleuMetric, RougeLMetric, WeightedEvaluator
from evaluation.retrieval_evaluator import RetrievalEvaluator
from generator.generator import ReplyGenerator
from generator.retriever import EmailRetriever

DATASET_PATH = "dataset/emails.json"
OFFLINE_METRIC_WEIGHTS = {"bleu": 0.5, "rouge_l": 0.5}
CONFIDENCE_THRESHOLD = 0.15

EXAMPLE_EMAILS = {
    "— pick an example —": "",
    "Shipping delay": "My package was supposed to arrive 4 days ago and tracking hasn't updated since. Where is it?",
    "Double charge": "I was charged twice for my subscription this month. Can you please look into this and refund the duplicate charge?",
    "Angry refund": "I returned my item three weeks ago and still haven't received my refund. This is really frustrating.",
    "Off-topic (tests low-confidence fallback)": "Do you sell gift wrapping for llamas? Asking for a friend's birthday party next Tuesday.",
}

st.set_page_config(page_title="ReplyIQ", page_icon="✉️", layout="wide")


@st.cache_resource
def load_pipeline():
    retriever = EmailRetriever(DATASET_PATH)
    generator = ReplyGenerator()
    offline_evaluator = WeightedEvaluator(
        metrics=[BleuMetric(), RougeLMetric()],
        weights=OFFLINE_METRIC_WEIGHTS,
    )
    judge = LlmJudgeMetric()
    retrieval_evaluator = RetrievalEvaluator()
    return retriever, generator, offline_evaluator, judge, retrieval_evaluator


retriever, generator, offline_evaluator, judge, retrieval_evaluator = load_pipeline()

with st.sidebar:
    st.header("✉️ ReplyIQ")
    st.markdown(
        "AI-suggested support replies, grounded in past emails, "
        "with a reference-free evaluation engine."
    )
    st.divider()
    st.markdown(
        "**How it works**\n\n"
        "1. TF-IDF retrieves the 3 most similar past emails\n"
        "2. A confidence check decides whether retrieval is trustworthy\n"
        "3. Llama 3.3 70B generates a grounded reply — or a safe fallback\n"
        "4. An LLM judge scores it on 6 dimensions, reference-free\n"
    )
    st.divider()
    st.caption("Judge rubric: correctness · helpfulness · completeness · professionalism · tone · faithfulness")
    st.caption("Fabricated claims gate the score to 0.4 regardless of other dimensions.")

st.title("ReplyIQ")
st.caption("Paste a customer email — retrieval, generation, and evaluation run live.")

example_choice = st.selectbox("Try an example", options=list(EXAMPLE_EMAILS.keys()))

customer_email = st.text_area(
    "Incoming customer email",
    value=EXAMPLE_EMAILS[example_choice],
    height=140,
    placeholder="e.g. My order arrived broken and I want a refund, not a replacement.",
)

with st.expander("Optional: historical reference reply (offline validation only)"):
    reference_reply = st.text_area(
        "Reference reply",
        height=100,
        label_visibility="collapsed",
        placeholder="Paste a known-good historical reply to also compute BLEU / ROUGE-L against it.",
    )

if st.button("Generate suggested reply", type="primary", use_container_width=True) and customer_email.strip():
    with st.status("Running pipeline...", expanded=True) as status:
        st.write("Retrieving similar past emails...")
        similar_examples = retriever.retrieve_similar(customer_email, top_k=3)
        confidence = retrieval_evaluator.confidence_score(similar_examples)
        low_confidence = confidence < CONFIDENCE_THRESHOLD

        st.write("Generating reply..." if not low_confidence else "Low confidence — generating safe fallback reply...")
        generated_reply = generator.generate_reply(customer_email, similar_examples, low_confidence=low_confidence)

        st.write("Scoring reply (reference-free judge)...")
        primary_result = judge.score(generated=generated_reply, reference="", customer_email=customer_email)

        offline_result = None
        if reference_reply.strip():
            st.write("Computing offline validation vs. reference...")
            offline_result = offline_evaluator.evaluate(generated_reply, reference_reply)

        status.update(label="Done", state="complete", expanded=False)

    if low_confidence:
        st.warning(
            f"**Low retrieval confidence ({confidence:.0%})** — no sufficiently similar historical "
            "emails found. Generated a safe, uncertainty-admitting reply instead of inventing specifics.",
            icon="⚠️",
        )

    left, right = st.columns([3, 2], gap="large")

    with left:
        st.subheader("Suggested reply")
        st.info(generated_reply)

        with st.expander(f"Retrieved grounding context (confidence: {confidence:.0%})"):
            for example in similar_examples:
                st.markdown(f"**{example['domain']}** · similarity {example['similarity_score']}")
                st.caption(f"Customer: {example['customer_email']}")
                st.caption(f"Reply: {example['support_reply']}")
                st.divider()

    with right:
        st.subheader("Evaluation")
        st.caption("Primary score is reference-free — what would run in production, where no ground truth exists.")

        score_cols = st.columns(2)
        score_cols[0].metric("Primary score", primary_result.score)
        score_cols[1].metric("Confidence", f"{confidence:.0%}")

        detail = json.loads(primary_result.detail)
        dims = ["correctness", "helpfulness", "completeness", "professionalism", "tone", "faithfulness"]
        dim_cols = st.columns(3)
        for i, dim in enumerate(dims):
            dim_cols[i % 3].metric(dim.capitalize(), f"{detail.get(dim, '—')}/10")

        if detail.get("weakness"):
            st.markdown(f"**Judge-identified weakness:** {detail['weakness']}")
        if detail.get("reasoning"):
            st.caption(detail["reasoning"])

        if offline_result:
            st.divider()
            st.caption("Offline validation (vs. provided reference — not available in production)")
            off_cols = st.columns(3)
            off_cols[0].metric("BLEU", offline_result["components"]["bleu"])
            off_cols[1].metric("ROUGE-L", offline_result["components"]["rouge_l"])
            off_cols[2].metric("Offline overall", offline_result["overall_score"])