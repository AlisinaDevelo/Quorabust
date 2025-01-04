import pytest

import quorabust.embedding_features as ef


def test_pair_embedding_builder_requires_sentence_transformers(monkeypatch):
    monkeypatch.setattr(ef, "SentenceTransformer", None)
    with pytest.raises(RuntimeError, match="nlp"):
        ef.PairEmbeddingBuilder()
