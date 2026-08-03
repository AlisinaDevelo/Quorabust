import pandas as pd

from quorabust.split import split_train_eval


def _frame_with_question_ids(n: int = 20) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "qid1": list(range(1, n * 2, 2)),
            "qid2": list(range(2, n * 2 + 1, 2)),
            "question1": [f"question {i}" for i in range(n)],
            "question2": [f"question {i + 1}" for i in range(n)],
            "is_duplicate": [i % 2 for i in range(n)],
        }
    )


def test_question_component_split_is_deterministic_and_disjoint():
    df = _frame_with_question_ids()

    train_a, eval_a, strategy_a = split_train_eval(df, eval_fraction=0.2, seed=7)
    train_b, eval_b, strategy_b = split_train_eval(df, eval_fraction=0.2, seed=7)

    assert strategy_a == strategy_b == "question_component_holdout"
    assert eval_a is not None and eval_b is not None
    pd.testing.assert_frame_equal(train_a, train_b)
    pd.testing.assert_frame_equal(eval_a, eval_b)

    train_questions = set(train_a["qid1"]) | set(train_a["qid2"])
    eval_questions = set(eval_a["qid1"]) | set(eval_a["qid2"])
    assert train_questions.isdisjoint(eval_questions)
    assert len(train_a) + len(eval_a) == len(df)


def test_question_component_split_keeps_transitive_questions_together():
    df = _frame_with_question_ids(24)
    df.loc[1, "qid1"] = df.loc[0, "qid2"]

    train, evaluation, strategy = split_train_eval(df, eval_fraction=0.25, seed=3)

    assert strategy == "question_component_holdout"
    assert evaluation is not None
    train_questions = set(train["qid1"]) | set(train["qid2"])
    eval_questions = set(evaluation["qid1"]) | set(evaluation["qid2"])
    assert train_questions.isdisjoint(eval_questions)


def test_split_falls_back_to_shuffled_rows_without_question_ids():
    df = _frame_with_question_ids(30).drop(columns=["qid1", "qid2"])

    train, evaluation, strategy = split_train_eval(df, eval_fraction=0.2, seed=11)

    assert strategy == "shuffled_prefix_holdout"
    assert evaluation is not None
    assert len(evaluation) == 6
    assert len(train) == 24


def test_split_can_disable_holdout_and_cap_rows():
    df = _frame_with_question_ids(30)

    train, evaluation, strategy = split_train_eval(
        df,
        eval_fraction=0,
        seed=11,
        max_rows=10,
    )

    assert strategy == "none"
    assert evaluation is None
    assert len(train) == 10
