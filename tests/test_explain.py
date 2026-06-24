import pandas as pd

from quorabust.explain import explain_pair_features, feature_names_for_builder
from quorabust.features import PairFeatureBuilder


def _builder():
    df = pd.DataFrame(
        {
            "question1": ["how to learn python", "best pizza"],
            "question2": ["python learning tips", "pizza places"],
        }
    )
    return PairFeatureBuilder(max_features=50).fit_from_frame(df)


def test_explain_pair_features_returns_named_values():
    builder = _builder()
    rows = explain_pair_features(
        builder,
        ["how to learn python"],
        ["python learning tips"],
    )
    assert set(rows[0]) == {"cos", "jaccard", "len_ratio", "abs_len_diff", "len_sum"}
    assert all(isinstance(v, float) for v in rows[0].values())


def test_feature_schema_overrides_builder_names():
    builder = _builder()
    assert feature_names_for_builder(builder, ["a", "b"]) == ["a", "b"]
