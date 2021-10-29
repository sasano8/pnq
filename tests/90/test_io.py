import tempfile

import pytest

import pnq
from pnq.io import from_csv
from tests.conftest import to_sync


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


@to_sync
async def test_to_file():
    from pnq._itertools._async import io as aio

    async def aiter():
        yield 1
        yield 2
        yield 3

    with tempfile.NamedTemporaryFile() as f:
        await aio.to_file(aiter(), f.name)
        assert f.read() == b"[1,2,3]"

    with tempfile.NamedTemporaryFile() as f:
        await aio.to_file(aiter(), f.name, "\n")
        assert f.read() == b"1\n2\n3"


@to_sync
async def test_to_json():
    from pnq._itertools._async import io as aio

    async def as_list():
        yield 1
        yield 2

    async def as_dict():
        yield ("a", 1)
        yield ("b", 2)

    with tempfile.NamedTemporaryFile() as f:
        await aio.to_json(as_list(), f.name, as_dict=False)
        assert f.read() == b"[\n  1,\n  2\n]\n"

    with tempfile.NamedTemporaryFile() as f:
        await aio.to_json(as_dict(), f.name, as_dict=True)
        assert f.read() == b'{\n  "a": 1,\n  "b": 2\n}\n'


@to_sync
async def test_to_jsonl():
    from pnq._itertools._async import io as aio

    async def aiter():
        yield {"a": 1}
        yield {"b": 2}

    with tempfile.NamedTemporaryFile() as f:
        await aio.to_jsonl(aiter(), f.name)
        assert f.read() == b'{"a": 1}\n{"b": 2}\n'


@to_sync
async def test_to_csv():
    from pnq._itertools._async import io as aio
    from pnq._itertools._async.queries import as_aiter

    ############
    # other mode
    ############
    with tempfile.NamedTemporaryFile() as f:
        await aio.to_csv(
            as_aiter([[1, "test"], [2, "test"]]),
            f.name,
            header=["id", "fullname"],
            skip_first=False,
        )
        assert list(from_csv(f.name)) == [
            ["id", "fullname"],
            ["1", "test"],
            ["2", "test"],
        ]

    with tempfile.NamedTemporaryFile() as f:
        await aio.to_csv(
            as_aiter([["id", "name"], [1, "test"]]),
            f.name,
            header=["id", "fullname"],
            skip_first=True,
        )
        assert list(from_csv(f.name)) == [["id", "fullname"], ["1", "test"]]

    with tempfile.NamedTemporaryFile() as f:
        await aio.to_csv(
            as_aiter([["id", "name"], [1, "test"]]),
            f.name,
            header=None,
            skip_first=False,
        )
        assert list(from_csv(f.name)) == [["id", "name"], ["1", "test"]]

    with tempfile.NamedTemporaryFile() as f:
        await aio.to_csv(
            as_aiter([["id", "name"], [1, "test"]]),
            f.name,
            header=True,
            skip_first=False,
        )
        assert list(from_csv(f.name)) == [["id", "name"], ["1", "test"]]

    with tempfile.NamedTemporaryFile() as f:
        await aio.to_csv(
            as_aiter([["id", "name"], [1, "test"]]),
            f.name,
            header=False,
            skip_first=False,
        )
        assert list(from_csv(f.name)) == [["id", "name"], ["1", "test"]]

    with tempfile.NamedTemporaryFile() as f:
        await aio.to_csv(
            as_aiter([["id", "name"], [1, "test"]]),
            f.name,
            header=None,
            skip_first=True,
        )
        assert list(from_csv(f.name)) == [["1", "test"]]

    with pytest.raises(ValueError, match="The behavior is undefined."):
        with tempfile.NamedTemporaryFile() as f:
            await aio.to_csv(
                as_aiter([["id", "name"], [1, "test"]]),
                f.name,
                header=True,
                skip_first=True,
            )
            assert list(from_csv(f.name)) == [["id", "name"], ["1", "test"]]

    with tempfile.NamedTemporaryFile() as f:
        await aio.to_csv(
            as_aiter([["id", "name"], [1, "test"]]),
            f.name,
            header=False,
            skip_first=True,
        )
        assert list(from_csv(f.name)) == [["1", "test"]]

    ############
    # dict mode
    ############
    with tempfile.NamedTemporaryFile() as f:
        await aio.to_csv(
            as_aiter([{"name": "n1", "age": 10}, {"name": "n2", "age": 20}]),
            f.name,
            header=True,
            skip_first=False,
        )
        assert list(from_csv(f.name)) == [["name", "age"], ["n1", "10"], ["n2", "20"]]

    with tempfile.NamedTemporaryFile() as f:
        await aio.to_csv(
            as_aiter([{"name": "n1", "age": 10}, {"name": "n2", "age": 20}]),
            f.name,
            header=None,
            skip_first=False,
        )
        assert list(from_csv(f.name)) == [["n1", "10"], ["n2", "20"]]

    with tempfile.NamedTemporaryFile() as f:
        await aio.to_csv(
            as_aiter([{"name": "n1", "age": 10}, {"name": "n2", "age": 20}]),
            f.name,
            header=["name", "age"],
            skip_first=False,
        )
        assert list(from_csv(f.name)) == [["name", "age"], ["n1", "10"], ["n2", "20"]]

    with tempfile.NamedTemporaryFile() as f:
        await aio.to_csv(
            as_aiter([{"name": "n1", "age": 10}, {"name": "n2", "age": 20}]),
            f.name,
            header=["name", "age"],
            skip_first=True,
        )
        assert list(from_csv(f.name)) == [["name", "age"], ["n2", "20"]]

    with pytest.raises(
        ValueError, match="dict contains fields not in fieldnames: 'age'"
    ):
        with tempfile.NamedTemporaryFile() as f:
            await aio.to_csv(
                as_aiter([{"name": "test", "age": 20}]),
                f.name,
                header=["name"],
                skip_first=False,
            )

    with tempfile.NamedTemporaryFile() as f:
        await aio.to_csv(
            as_aiter([{"name": "test", "age": 20}]),
            f.name,
            header=["name"],
            skip_first=True,
        )
        assert list(from_csv(f.name)) == [["name"]]
