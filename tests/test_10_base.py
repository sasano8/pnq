import asyncio
from functools import wraps

import pytest

from pnq.base.core import IterType, Query, QueryNormal, QuerySyncToAsync
from pnq.base.queries import Map, AsyncMap


def async_test(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        async def main():
            return await func(*args, **kwargs)

        return asyncio.run(main())

    return wrapper


async def create_aiter():
    yield 4
    yield 5
    yield 6


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
