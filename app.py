import streamlit as st

from evaluation.judge import LlmJudgeMetric
from evaluation.metrics import BertScoreMetric, BleuMetric, RougeLMetric, WeightedEvaluator
from generator.generator import ReplyGenerator
from generator.retriever import EmailRetriever

DATASET_PATH = "dataset/emails.json"
METRIC_WEIGHTS = {"bleu": 0.15, "rouge_l": 0.15, "bert_score": 0.30, "llm_judge": 0.40}

st.set_page_config(page_title="ReplyIQ", layout="wide")


@st.cache_resource
def load_pipeline():
    retriever = EmailRetriever(DATASET_PATH)
    generator = ReplyGenerator(model_name="gemini-3.5-flash")
    lexical_evaluator = WeightedEvaluator(
        metrics=[BleuMetric(), RougeLMetric(), BertScoreMetric()],
        weights=METRIC_WEIGHTS,
    )
    judge = LlmJudgeMetric(model_name="gemini-3.5-flash")
    return retriever, generator, lexical_evaluator, judge


st.title("ReplyIQ — Live Suggested Reply Demo")
st.caption("Paste a real customer email below. Retrieval, generation, and scoring all run live.")

retriever, generator, lexical_evaluator, judge = load_pipeline()

customer_email = st.text_area(
    "Incoming customer email",
    height=140,
    placeholder="e.g. My order arrived broken and I want a refund, not a replacement.",
)

reference_reply = st.text_area(
    "Optional: reference reply to score against (leave blank to skip scoring)",
    height=100,
)

if st.button("Generate suggested reply", type="primary") and customer_email.strip():
    with st.spinner("Retrieving similar past emails..."):
        similar_examples = retriever.retrieve_similar(customer_email, top_k=3)

    with st.expander("Retrieved similar emails (grounding context)"):
        for example in similar_examples:
            st.markdown(f"**Similarity: {example['similarity_score']}** — *{example['domain']}*")
            st.write(f"Customer: {example['customer_email']}")
            st.write(f"Reply: {example['support_reply']}")
            st.divider()

    with st.spinner("Generating reply with Gemini..."):
        generated_reply = generator.generate_reply(customer_email, similar_examples)

    st.subheader("Suggested reply")
    st.info(generated_reply)

    if reference_reply.strip():
        with st.spinner("Scoring reply..."):
            lexical_result = lexical_evaluator.evaluate(generated_reply, reference_reply)
            judge_result = judge.score(generated_reply, reference_reply, customer_email)
            overall_score = round(
                lexical_result["overall_score"] + judge_result.score * METRIC_WEIGHTS["llm_judge"],
                4,
            )

        st.subheader("Evaluation")
        cols = st.columns(5)
        cols[0].metric("BLEU", lexical_result["components"]["bleu"])
        cols[1].metric("ROUGE-L", lexical_result["components"]["rouge_l"])
        cols[2].metric("BERTScore", lexical_result["components"]["bert_score"])
        cols[3].metric("LLM Judge", judge_result.score)
        cols[4].metric("Overall", overall_score)

        with st.expander("Judge reasoning"):
            st.json(judge_result.detail)