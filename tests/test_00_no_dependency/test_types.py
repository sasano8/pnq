# type: ignore
import asyncio

import pnq


def test_args():
    x1 = pnq.args()
    assert [*x1] == []
    assert {**x1} == {}
    assert x1.args == tuple()
    assert x1.kwargs == {}
    x2 = pnq.args(1, a=2)
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
