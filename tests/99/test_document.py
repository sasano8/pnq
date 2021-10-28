import asyncio

import pnq
from pnq.concurrent import AsyncPool, ProcessPool, ThreadPool


def mul(x):
    return x * 2


async def mul_async(x):
    return x * 2


async def aiter():
    yield 1
    yield 2
    yield 3


async def main():
    async with ProcessPool(2) as proc, ThreadPool(2) as thread, AsyncPool(2) as aio:
        tasks = pnq.query(
            [
                pnq.query([1, 2, 3]).parallel(mul, proc),
                pnq.query([1, 2, 3]).parallel(mul_async, proc),
                pnq.query(aiter()).parallel(mul, proc),
                pnq.query(aiter()).parallel(mul_async, proc),
                pnq.query([1, 2, 3]).parallel(mul, thread),
                pnq.query([1, 2, 3]).parallel(mul_async, thread),
                pnq.query(aiter()).parallel(mul, thread),
                pnq.query(aiter()).parallel(mul_async, thread),
                pnq.query([1, 2, 3]).parallel(mul, aio),
                pnq.query([1, 2, 3]).parallel(mul_async, aio),
                pnq.query(aiter()).parallel(mul, aio),
                pnq.query(aiter()).parallel(mul_async, aio),
            ]
        )

        assert tasks.len() == 12
        results = await tasks.gather()
        assert results.len() == 12
        for x in results:
            assert x == [2, 4, 6]


def test_document():
    asyncio.run(main())
