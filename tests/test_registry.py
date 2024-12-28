from quorabust.registry import append_model_record, load_model_records


def test_append_and_load_roundtrip(tmp_path):
    d = tmp_path / "reg"
    append_model_record(d, {"model_id": "m1", "path": "/tmp/a.pkl"})
    append_model_record(d, {"model_id": "m2", "path": "/tmp/b.pkl"})
    rows = load_model_records(d)
    assert len(rows) == 2
    assert rows[0]["model_id"] == "m1"
    assert rows[1]["model_id"] == "m2"
    assert "registered_at" in rows[0]
