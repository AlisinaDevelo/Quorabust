"""Quorabust: text preprocessing and pairwise features for duplicate questions."""

from importlib.metadata import PackageNotFoundError, version

from quorabust.features import PairFeatureBuilder, word_jaccard
from quorabust.model import eval_classification_metrics, train_duplicate_classifier
from quorabust.persist import load_classifier, save_classifier
from quorabust.preprocess import clean_text
from quorabust.protocol_builder import build_protocol, build_protocol_payload
from quorabust.retrieval import (
    CatalogHit,
    CatalogQuestion,
    SentenceTransformerCatalogRetriever,
    SentenceTransformerCrossEncoderReranker,
    TfidfCatalogRetriever,
)
from quorabust.retrieval_benchmark import (
    RetrievalCase,
    benchmark_retrieval,
    evaluate_rankings,
    query_length_bucket,
)
from quorabust.safe_artifact import load_safe_classifier, safe_metadata, save_safe_classifier

try:
    __version__ = version("Quorabust")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = [
    "__version__",
    "clean_text",
    "word_jaccard",
    "PairFeatureBuilder",
    "train_duplicate_classifier",
    "eval_classification_metrics",
    "save_classifier",
    "load_classifier",
    "CatalogQuestion",
    "CatalogHit",
    "TfidfCatalogRetriever",
    "SentenceTransformerCatalogRetriever",
    "SentenceTransformerCrossEncoderReranker",
    "RetrievalCase",
    "benchmark_retrieval",
    "evaluate_rankings",
    "query_length_bucket",
    "save_safe_classifier",
    "load_safe_classifier",
    "safe_metadata",
    "build_protocol",
    "build_protocol_payload",
]
