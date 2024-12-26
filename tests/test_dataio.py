import pandas as pd

from quorabust.dataio import iter_csv_batches


def test_iter_csv_batches_yields_rows(tmp_path):
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    p = tmp_path / "x.csv"
    df.to_csv(p, index=False)
    chunks = list(iter_csv_batches(p, chunksize=2))
    assert len(chunks) == 2
    assert len(chunks[0]) == 2
    assert len(chunks[1]) == 1
