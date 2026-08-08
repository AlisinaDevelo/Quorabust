import json

import pandas as pd

from quorabust.hard_negative_cli import main
from quorabust.lineage import sha256_file


def _write_input(path):
    pd.DataFrame(
        {
            "question1": [
                "How do I deploy Python services?",
                "How can I deploy Python applications?",
                "How do I deploy Python services?",
                "What is the weather today?",
            ],
            "question2": [
                "How can I deploy Python applications?",
                "What is the best way to deploy Python code?",
                "How do I deploy Python applications cheaply?",
                "Will it rain tomorrow?",
            ],
            "is_duplicate": [1, 1, 0, 0],
            "qid1": ["q1", "q2", "q1", "q5"],
            "qid2": ["q2", "q3", "q4", "q6"],
        }
    ).to_csv(path, index=False)


def test_cli_writes_training_pairs_and_hash_bound_sidecar(tmp_path):
    source = tmp_path / "train.csv"
    output = tmp_path / "hard-negatives.csv"
    metadata = tmp_path / "hard-negatives.meta.json"
    _write_input(source)

    assert (
        main(
            [
                "--csv",
                str(source),
                "--out",
                str(output),
                "--metadata-out",
                str(metadata),
                "--candidate-k",
                "6",
                "--negatives-per-positive",
                "2",
            ]
        )
        == 0
    )

    generated = pd.read_csv(output)
    payload = json.loads(metadata.read_text(encoding="utf-8"))
    assert not generated.empty
    assert set(generated["is_duplicate"]) == {0}
    assert payload["source"]["sha256"] == sha256_file(source)
    assert payload["output"]["sha256"] == sha256_file(output)
    assert payload["config"]["retriever"] == "tfidf"
    assert payload["safeguards"]["final_holdout_used"] is False


def test_cli_refuses_to_overwrite_generated_artifacts(tmp_path, capsys):
    source = tmp_path / "train.csv"
    output = tmp_path / "hard-negatives.csv"
    _write_input(source)

    arguments = ["--csv", str(source), "--out", str(output)]
    assert main(arguments) == 0
    assert main(arguments) == 1
    assert "refusing to overwrite" in capsys.readouterr().err


def test_cli_accepts_embedding_backend_configuration(tmp_path, monkeypatch):
    source = tmp_path / "train.csv"
    output = tmp_path / "hard-negatives.csv"
    metadata = tmp_path / "hard-negatives.meta.json"
    _write_input(source)

    import quorabust.hard_negatives as hard_negatives

    class FakeDenseRetriever:
        def __init__(self, model_name):
            assert model_name == "fake-dense"
            self._questions = []

        @property
        def size(self):
            return len(self._questions)

        def fit(self, questions):
            self._questions = list(questions)
            return self

        def search(self, _query, *, k):
            from quorabust.retrieval import CatalogHit

            return [
                CatalogHit("q1", "How do I deploy Python services?", 0.99),
                CatalogHit("q4", "How do I deploy Python applications cheaply?", 0.88),
            ][:k]

    monkeypatch.setattr(hard_negatives, "SentenceTransformerCatalogRetriever", FakeDenseRetriever)
    assert (
        main(
            [
                "--csv",
                str(source),
                "--out",
                str(output),
                "--metadata-out",
                str(metadata),
                "--retriever",
                "embedding",
                "--embedding-model",
                "fake-dense",
            ]
        )
        == 0
    )

    payload = json.loads(metadata.read_text(encoding="utf-8"))
    assert payload["config"]["retriever"] == "embedding"
    assert payload["config"]["model_name"] == "fake-dense"
