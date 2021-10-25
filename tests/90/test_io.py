import tempfile

import pnq


def test_csv():
    with tempfile.NamedTemporaryFile() as f:
        f.write(b"name,age\n")
        f.write(b"bob,20")
        f.flush()

        with pnq.io.from_csv(f.name) as lines:
            assert lines.first() == ["name", "age"]
            assert lines.first() == ["bob", "20"]


def test_jsonl():
    with tempfile.NamedTemporaryFile() as f:
        f.write(b"\n")
        f.write(b"")
        f.flush()
        with pnq.io.from_jsonl(f.name) as lines:
            assert lines.result() == []

    with tempfile.NamedTemporaryFile() as f:
        f.write(b'{"a": 1}\n')
        f.write(b'{"b": 2}')
        f.flush()

        with pnq.io.from_jsonl(f.name) as lines:
            assert lines.first() == {"a": 1}
            assert lines.first() == {"b": 2}
