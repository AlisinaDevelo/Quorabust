import pandas as pd
import pytest

from quorabust.hard_negatives import mine_hard_negatives


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
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
    )


def test_mines_ranked_negatives_outside_known_positive_component():
    result = mine_hard_negatives(_frame(), candidate_k=6, negatives_per_positive=2)

    assert not result.pairs.empty
    assert set(result.pairs["is_duplicate"]) == {0}
    assert set(result.pairs.columns) == {
        "question1",
        "question2",
        "is_duplicate",
        "qid1",
        "qid2",
        "source_positive_row",
        "source_positive_qid1",
        "source_positive_qid2",
        "anchor_side",
        "retrieval_rank",
        "retrieval_score",
    }
    for row in result.pairs.itertuples(index=False):
        assert row.qid2 not in {"q1", "q2", "q3"}
        assert row.anchor_side in {"qid1", "qid2"}
        assert row.retrieval_score >= 0.0
    assert result.metadata["safeguards"]["anchor_positive_component_excluded"] is True
    assert result.metadata["output"]["rows"] == len(result.pairs)


def test_mining_is_byte_stable_for_same_configuration():
    first = mine_hard_negatives(
        _frame(),
        candidate_k=6,
        negatives_per_positive=2,
        max_positive_rows=1,
        seed=7,
    )
    second = mine_hard_negatives(
        _frame(),
        candidate_k=6,
        negatives_per_positive=2,
        max_positive_rows=1,
        seed=7,
    )

    pd.testing.assert_frame_equal(first.pairs, second.pairs)
    assert first.metadata == second.metadata


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda frame: frame.drop(columns=["qid1"]), "requires columns"),
        (lambda frame: frame.assign(qid1=["q1", "", "q1", "q5"]), "must not be empty"),
        (lambda frame: frame.assign(is_duplicate=[1, 2, 0, 0]), "binary"),
        (lambda frame: frame.iloc[0:0], "at least one row"),
    ],
)
def test_mining_fails_closed_for_invalid_inputs(mutator, message):
    with pytest.raises(ValueError, match=message):
        mine_hard_negatives(mutator(_frame()))


def test_mining_rejects_unsafe_parameters():
    with pytest.raises(ValueError, match="candidate_k"):
        mine_hard_negatives(_frame(), candidate_k=0)
    with pytest.raises(ValueError, match="negatives_per_positive"):
        mine_hard_negatives(_frame(), negatives_per_positive=0)
    with pytest.raises(ValueError, match="max_positive_rows"):
        mine_hard_negatives(_frame(), max_positive_rows=0)
    with pytest.raises(ValueError, match="seed"):
        mine_hard_negatives(_frame(), seed=-1)


def test_mining_rejects_conflicting_question_text():
    frame = _frame()
    frame.loc[1, "question1"] = "A conflicting version of q2"

    with pytest.raises(ValueError, match="conflicting text"):
        mine_hard_negatives(frame)


def test_mining_requires_an_eligible_component_outside_the_anchor():
    frame = _frame().iloc[[0]].copy()

    with pytest.raises(ValueError, match="no eligible hard negatives"):
        mine_hard_negatives(frame)
