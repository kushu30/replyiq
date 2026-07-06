import json

import pytest

from generator.retriever import EmailRetriever


@pytest.fixture
def sample_dataset(tmp_path):
    data = [
        {"id": "1", "domain": "refunds", "customer_email": "I want a refund for my broken order",
         "support_reply": "Refund processed."},
        {"id": "2", "domain": "billing", "customer_email": "I was charged twice this month",
         "support_reply": "Duplicate charge refunded."},
        {"id": "3", "domain": "refunds", "customer_email": "My item arrived damaged, refund please",
         "support_reply": "Refund issued for damaged item."},
    ]
    path = tmp_path / "emails.json"
    path.write_text(json.dumps(data))
    return str(path)


def test_retrieve_similar_returns_requested_count(sample_dataset):
    retriever = EmailRetriever(sample_dataset)
    results = retriever.retrieve_similar("My package arrived broken, I want money back", top_k=2)
    assert len(results) == 2


def test_retrieve_similar_ranks_relevant_domain_higher(sample_dataset):
    retriever = EmailRetriever(sample_dataset)
    results = retriever.retrieve_similar("My package arrived broken, I want a refund", top_k=1)
    assert results[0]["domain"] == "refunds"


def test_retrieve_similar_includes_similarity_score(sample_dataset):
    retriever = EmailRetriever(sample_dataset)
    results = retriever.retrieve_similar("refund for damaged order", top_k=1)
    assert "similarity_score" in results[0]
    assert 0.0 <= results[0]["similarity_score"] <= 1.0


def test_missing_dataset_raises_file_not_found():
    with pytest.raises(FileNotFoundError):
        EmailRetriever("does/not/exist.json")