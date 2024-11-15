"""Quorabust: text preprocessing and pairwise features for duplicate questions."""

from importlib.metadata import PackageNotFoundError, version

from quorabust.features import PairFeatureBuilder, word_jaccard
from quorabust.model import eval_classification_metrics, train_duplicate_classifier
from quorabust.persist import load_classifier, save_classifier
from quorabust.preprocess import clean_text

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
]
