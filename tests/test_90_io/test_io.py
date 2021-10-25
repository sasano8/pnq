import tempfile

import pnq


def test_jsonl():
    with tempfile.NamedTemporaryFile() as f:
        f.write(b"\n")
        f.write(b"")
        f.flush()

        assert pnq.io.Jsonl(f.name).result() == []

    with tempfile.NamedTemporaryFile() as f:
        f.write(b'{"a": 1}')
        f.flush()

        assert pnq.io.Jsonl(f.name).result() == [{"a": 1}]
