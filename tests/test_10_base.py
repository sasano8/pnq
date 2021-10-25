import asyncio
from functools import wraps

import pytest

from pnq._itertools import AsyncMap, DebugPath, Lazy, Map, Sleep

# from pnq.base.core import IterType, Query, QueryNormal, QuerySyncToAsync
# from pnq.base.queries import AsyncMap, DebugPath, Lazy, Map, Sleep
from pnq._itertools.core import IterType, Query, QueryNormal, QuerySyncToAsync

# import pnq


def async_test(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        async def main():
            return await func(*args, **kwargs)

        return asyncio.run(main())

    return wrapper


async def aiter(iterable=[]):
    for i in iterable:
        yield i


async def create_aiter():
    yield 4
    yield 5
    yield 6


async def list_async(aiter):
    return [x async for x in aiter]


def test_iter_type():
    assert not IterType.IMPOSSIBLE
    assert IterType.NORMAL
    assert IterType.ASYNC
    assert IterType.BOTH

    assert (IterType.NORMAL & IterType.BOTH) == IterType.NORMAL
    assert (IterType.ASYNC & IterType.BOTH) == IterType.ASYNC
    assert (IterType.BOTH & IterType.BOTH) == IterType.BOTH

    assert (IterType.NORMAL & IterType.ASYNC) == IterType.IMPOSSIBLE
    assert (IterType.ASYNC & IterType.ASYNC) == IterType.ASYNC
    assert (IterType.BOTH & IterType.ASYNC) == IterType.ASYNC

    assert (IterType.NORMAL & IterType.NORMAL) == IterType.NORMAL
    assert (IterType.ASYNC & IterType.NORMAL) == IterType.IMPOSSIBLE
    assert (IterType.BOTH & IterType.NORMAL) == IterType.NORMAL

    assert (IterType.NORMAL & IterType.IMPOSSIBLE) == IterType.IMPOSSIBLE
    assert (IterType.ASYNC & IterType.IMPOSSIBLE) == IterType.IMPOSSIBLE
    assert (IterType.BOTH & IterType.IMPOSSIBLE) == IterType.IMPOSSIBLE


class Test010_Async:
    @async_test
    async def test_async(self):
        q1 = Query([1, 2, 3])
        assert list(q1) == [1, 2, 3]

        with pytest.raises(NotImplementedError):
            [x async for x in q1]

        q2 = Query(create_aiter())
        assert [x async for x in q2] == [4, 5, 6]

        with pytest.raises(NotImplementedError):
            list(q2)

    @async_test
    async def test_chain(self):
        q1 = QueryNormal([1, 2, 3])
        assert list(q1) == [1, 2, 3]
        assert [x async for x in q1] == [1, 2, 3]

        q2 = Query(q1)
        assert list(q2) == [1, 2, 3]
        assert [x async for x in q2] == [1, 2, 3]

        q3 = QuerySyncToAsync(q2)
        with pytest.raises(NotImplementedError):
            assert list(q3) == [1, 2, 3]

        assert [x async for x in q3] == [1, 2, 3]

        q4 = Map(q1, lambda x: x * 2)
        assert list(q4) == [2, 4, 6]
        assert [x async for x in q4] == [2, 4, 6]

        q5 = Map(q2, lambda x: x * 2)
        assert list(q5) == [2, 4, 6]
        assert [x async for x in q5] == [2, 4, 6]

        q6 = Map(q3, lambda x: x * 2)
        with pytest.raises(NotImplementedError):
            assert list(q6) == [2, 4, 6]
        assert [x async for x in q6] == [2, 4, 6]

        q7 = Map(q6, lambda x: x * 2)
        with pytest.raises(NotImplementedError):
            assert list(q7) == [4, 8, 12]
        assert [x async for x in q7] == [4, 8, 12]

        async def mul_2(x):
            return x * 2

        q8 = AsyncMap(q7, mul_2)
        with pytest.raises(NotImplementedError):
            assert list(q8) == [8, 16, 24]
        assert [x async for x in q8] == [8, 16, 24]

        q9 = Map(q8, lambda x: x * 2)
        with pytest.raises(NotImplementedError):
            assert list(q9) == [16, 32, 48]
        assert [x async for x in q9] == [16, 32, 48]

    @async_test
    async def test_lazy(self):
        assert Lazy(Query([1, 2, 3]), list)() == [1, 2, 3]
        assert await Lazy(Query(create_aiter()), list_async) == [4, 5, 6]

    @async_test
    async def test_map_debug(self):
        map_five = lambda x: 5  # noqa
        map_ten = lambda x: 10  # noqa
        assert Lazy(DebugPath(Query([1, 2, 3]), map_five, map_ten), list)() == [5, 5, 5]
        assert await Lazy(
            DebugPath(Query(create_aiter()), map_five, map_ten), list_async
        ) == [10, 10, 10]

    @async_test
    async def test_sleep(self):
        assert list(Sleep(Query([1, 2, 3]), 0)) == [1, 2, 3]
        assert [x async for x in Sleep(Query(aiter([1, 2, 3])), 0)] == [1, 2, 3]

    def test_builder(self):
        # from pnq.base.builder import Builder
        from pnq._itertools.builder import Builder

        assert list(Builder.query([1, 2, 3])) == [1, 2, 3]
        assert list(Builder.query({1: "a", 2: "b"})) == [(1, "a"), (2, "b")]

    def test_builder_run(self):
        import tempfile

        # from pnq.base.builder import Builder
        # from pnq.base.requests import CancelToken
        from pnq._itertools.builder import Builder
        from pnq._itertools.requests import CancelToken

        async def func_1():
            return 100

        assert Builder.run(func_1) == 100

        with tempfile.TemporaryFile() as f:
            # 引数なしで関数が実行できること
            async def write_1():
                f.write(b"test1")
                f.flush()

            Builder.run(write_1)
            f.seek(0)
            assert f.read() == b"test1"

        with tempfile.TemporaryFile() as f:
            # tokenを受け取ってファイルを書き込めること
            async def write_2(token):
                f.write(b"test2")
                f.flush()
                assert isinstance(token, CancelToken)
                assert token.is_running()

            Builder.run(write_2)
            f.seek(0)
            assert f.read() == b"test2"

    @pytest.mark.slow
    def test_builder_run_signal(self):
        import signal
        import subprocess
        import time
        from functools import partial

        popen = partial(
            subprocess.Popen, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

        p = popen(["python", "tests/__signal_test", "0"])
        outs, errs = p.communicate()
        assert b"" in outs
        assert b"" in outs
        assert p.returncode == 0

        p = popen(["python", "tests/__signal_test", "1"])
        outs, errs = p.communicate()
        assert b"" in outs
        assert b"" in outs
        assert p.returncode == 0

        # SIGINTを送信した時、強制キャンセルされることを確認
        p = popen(["python", "tests/__signal_test", "0"])
        time.sleep(0.1)
        p.send_signal(signal.SIGINT)
        outs, errs = p.communicate()
        assert b"SIGINT" in outs
        assert b"forcibly" in outs
        assert b"Task was destroyed" in errs
        assert p.returncode == 1

        # SIGINTを送信した時、強制キャンセルされないことを確認
        p = popen(["python", "tests/__signal_test", "1"])
        time.sleep(0.1)
        p.send_signal(signal.SIGINT)
        outs, errs = p.communicate()
        assert b"SIGINT" in outs
        assert b"safely shut down" in outs
        assert b"" in errs
        assert p.returncode == 0
