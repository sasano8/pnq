# type: ignore
import asyncio

import pnq


def test_arguments():
    x1 = pnq.Arguments()
    assert [*x1] == []
    assert {**x1} == {}
    assert x1.args == tuple()
    assert x1.kwargs == {}
    x2 = pnq.Arguments(1, a=2)
    assert [*x2] == [1]
    assert {**x2} == {"a": 2}
    assert x2.args == (1,)
    assert x2.kwargs == {"a": 2}


def test_exitstack():
    class Hoge:
        def __init__(self):
            self.state = 0

        def __enter__(self):
            self.state = 1
            return self

        def __exit__(self, *args, **kwargs):
            self.state = 2

        async def __aenter__(self):
            self.state = 3
            return self

        async def __aexit__(self, *args, **kwargs):
            self.state = 4

    async def main():
        hoge = Hoge()
        assert hoge.state == 0
        with pnq.exitstack(hoge) as ctx:
            assert ctx[0] == hoge
            assert hoge.state == 1

        assert hoge.state == 2

        hoge = Hoge()
        assert hoge.state == 0
        async with pnq.exitstack(hoge) as ctx:
            assert ctx[0] == hoge
            assert hoge.state == 3

        assert hoge.state == 4

    asyncio.run(main())


def test_buffer():
    from pnq.io import BufferCache, GeneratorReader

    cache = BufferCache()
    cache.write(b"abc")

    assert cache.size == 3

    buf = cache.read(1)
    assert cache.size == 2
    assert buf == b"a"

    buf = cache.read(2)
    assert cache.size == 0
    assert buf == b"bc"

    buf = cache.read(0)
    assert cache.size == 0
    assert buf == b""

    buf = cache.read()
    assert cache.size == 0
    assert buf == b""

    buf = cache.read(3)
    assert cache.size == 0
    assert buf == b""

    cache.write(b"abc")
    buf = cache.read()
    assert cache.size == 0
    assert buf == b"abc"

    cache.write(b"abc")
    buf = cache.read(10)
    assert cache.size == 0
    assert buf == b"abc"

    cache.write(b"")
    assert list(cache.defrag(1)) == []
    assert cache.size == 0

    cache.write(b"a")
    assert list(cache.defrag(1)) == [b"a"]
    assert cache.size == 0

    cache.write(b"ab")
    assert list(cache.defrag(1)) == [b"a", b"b"]
    assert cache.size == 0

    cache.write(b"a")
    assert list(cache.defrag(2)) == [b"a"]
    assert cache.size == 0

    cache.write(b"ab")
    assert list(cache.defrag(2)) == [b"ab"]
    assert cache.size == 0

    cache.write(b"abc")
    assert list(cache.defrag(2)) == [b"ab", b"c"]
    assert cache.size == 0

    cache.write(b"abc")
    assert list(cache.defrag(2, keep_remain=True)) == [b"ab"]
    assert cache.size == 1

    def defrag(iterator, size: int):
        if size < 1:
            raise ValueError("size must be greater than 0.")

        cache = BufferCache()
        for val in iterator:
            cache.write(val)
            yield from cache.defrag(size, keep_remain=True)

        yield from cache.defrag(size, keep_remain=False)

    fragments = list(defrag(["abc", "de", "fghi", "jklm"], 2))
    assert fragments == ["ab", "cd", "ef", "gh", "ij", "kl", "m"]

    fragments = list(defrag([], 2))
    assert fragments == []

    assert GeneratorReader(b"", []).read() == b""
    assert GeneratorReader(b"", [b"a", b"b"]).read() == b"ab"
    assert GeneratorReader(b"", [b"ab", b"cde"]).read() == b"abcde"
    assert GeneratorReader(b"", [b"ab", b"cde"]).read(1) == b"a"
    assert GeneratorReader(b"", [b"ab", b"cde"]).read(2) == b"ab"
    assert GeneratorReader(b"", [b"ab", b"cde"]).read(3) == b"abc"
    assert GeneratorReader(b"", [b"ab", b"cde"]).read(4) == b"abcd"
    assert GeneratorReader(b"", [b"ab", b"cde"]).read(5) == b"abcde"
