import asyncio
from typing import List

import pnq
from pnq import concurrent
from pnq._itertools.requests import Response
from tests.conftest import to_sync


def mul(x):
    return x * 2


async def mul_async(x):
    return x * 2


def add_one(x: int):
    return x + 1


async def add_two(x: int):
    return x + 2


class Test600_Concurrent:
    @to_sync
    async def test_parallel_processpool(self):
        from pnq.concurrent import ProcessPool as Pool

        with Pool(1) as pool:
            assert (
                pnq.query([1])
                .parallel(add_one, executor=pool)
                .parallel(add_two, executor=pool)
                .result()
            ) == [4]

            assert (
                await pnq.query([1])
                .parallel(add_one, executor=pool)
                .parallel(add_two, executor=pool)
            ) == [4]

    @to_sync
    async def test_parallel_threadpool(self):
        from pnq.concurrent import ThreadPool as Pool

        with Pool(1) as pool:
            assert (
                pnq.query([1])
                .parallel(add_one, executor=pool)
                .parallel(add_two, executor=pool)
                .result()
            ) == [4]

            assert (
                await pnq.query([1])
                .parallel(add_one, executor=pool)
                .parallel(add_two, executor=pool)
            ) == [4]

    @to_sync
    async def test_parallel_asyncpool(self):
        from pnq.concurrent import AsyncPool as Pool

        async with Pool(1) as pool:
            # assert (
            #     pnq([1])
            #     .parallel(add_one, executor=pool)
            #     .parallel(add_two, executor=pool)
            #     .result()
            # ) == [4]

            assert (
                await pnq.query([1])
                .parallel(add_one, executor=pool)
                .parallel(add_two, executor=pool)
            ) == [4]

    @to_sync
    async def test_parallel_dummypool(self):
        from pnq.concurrent import DummyPool as Pool

        with Pool(1) as pool:
            assert (
                pnq.query([1]).parallel(add_one, executor=pool)
                # .parallel(add_two, executor=pool)
                .result()
            ) == [2]

            assert (
                await pnq.query([1])
                .parallel(add_one, executor=pool)
                .parallel(add_two, executor=pool)
            ) == [4]

    @to_sync
    async def test_parallel_dummypool_default(self):
        assert (
            pnq.query([1]).parallel(add_one)
            # .parallel(add_two, executor=pool)
            .result()
        ) == [2]

        assert (await pnq.query([1]).parallel(add_one).parallel(add_two)) == [4]

    @to_sync
    async def test_dispatch_threadpool(self):
        from pnq.concurrent import ThreadPool as Pool

        with Pool(1) as pool:
            assert (
                pnq.query([1])
                .parallel(add_one, executor=pool)
                .parallel(add_two, executor=pool)
                .result()
            ) == [4]

            assert (
                await pnq.query([1])
                .parallel(add_one, executor=pool)
                .parallel(add_two, executor=pool)
            ) == [4]

    @to_sync
    async def test_dispatch_asyncpool(self):
        from pnq.concurrent import AsyncPool as Pool

        async with Pool(1) as pool:
            # assert (
            #     pnq([1])
            #     .parallel(add_one, executor=pool)
            #     .parallel(add_two, executor=pool)
            #     .result()
            # ) == [4]

            assert (
                await pnq.query([1])
                .parallel(add_one, executor=pool)
                .parallel(add_two, executor=pool)
            ) == [4]

    @to_sync
    async def test_dispatch_dummypool(self):
        from pnq.concurrent import DummyPool as Pool

        with Pool(1) as pool:
            assert (
                pnq.query([1]).parallel(add_one, executor=pool)
                # .parallel(add_two, executor=pool)
                .result()
            ) == [2]

            assert (
                await pnq.query([1])
                .parallel(add_one, executor=pool)
                .parallel(add_two, executor=pool)
            ) == [4]

    @to_sync
    async def test_dispatch_dummypool_default(self):
        assert (
            pnq.query([1]).parallel(add_one)
            # .parallel(add_two, executor=pool)
            .result()
        ) == [2]

        assert (await pnq.query([1]).parallel(add_one).parallel(add_two)) == [4]


@to_sync
async def test_pool():
    async with concurrent.ProcessPool(1) as pool:
        result1 = (await pnq.query([1]).request(mul, pool))[0]
        result2 = (await pnq.query([1]).request(mul_async, pool))[0]

    async with concurrent.ThreadPool(1) as pool:
        result3 = (await pnq.query([1]).request(mul, pool))[0]
        result4 = (await pnq.query([1]).request(mul_async, pool))[0]

    async with concurrent.AsyncPool(1) as pool:
        result5 = (await pnq.query([1]).request(mul, pool))[0]
        result6 = (await pnq.query([1]).request(mul_async, pool))[0]

    async with concurrent.DummyPool(1) as pool:
        result7 = (await pnq.query([1]).request(mul, pool))[0]
        result8 = (await pnq.query([1]).request(mul_async, pool))[0]

    results = [
        result1,
        result2,
        result3,
        result4,
        result5,
        result6,
        result7,
        result8,
    ]

    for i, x in enumerate(results):
        assert isinstance(x, Response)
        assert x.result == 2
        assert x.err is None


def test_request():
    from datetime import datetime

    from pnq._itertools.requests import Response

    result = []

    def ok(value):
        result.append(value)
        return "ok"

    async def ok_async(value):
        result.append(value)
        return "ok"

    def err(value1, value2):
        raise Exception("error")

    async def err_async(value1, value2):
        raise Exception("error")

    def main(ok, err):
        result.clear()

        response: List[Response] = (
            pnq.query([{"value": 1}]).request(ok, unpack="**").to(list)
        )
        assert result == [1]
        res = response[0]
        assert res.func == ok
        assert res.args == tuple()
        assert res.kwargs == {"value": 1}
        assert res.err is None
        assert res.result == "ok"
        assert isinstance(res.start, datetime)
        assert isinstance(res.end, datetime)

        response: List[Response] = (
            pnq.query([{"value1": 1, "value2": 2}]).request(err, unpack="**").to(list)
        )
        assert result == [1]
        res = response[0]
        assert res.func == err
        assert res.args == tuple()
        assert res.kwargs == {"value1": 1, "value2": 2}
        assert str(res.err) == "error"
        assert res.result is None
        assert isinstance(res.start, datetime)
        assert isinstance(res.end, datetime)

        return True

    assert main(ok, err)
    assert main(ok_async, err_async)


def test_dispatch():
    succeeded = []
    failed = []

    def ok(x):
        return x

    async def ok_async(x):
        return x

    def err(x):
        raise Exception("error")

    async def err_async(x):
        raise Exception("error")

    def on_complete(future):
        try:
            succeeded.append(future.result())
        except Exception as e:
            failed.append(e)

    def main():
        succeeded.clear()

        pnq.query([1, 2]).map(lambda x: x).dispatch(ok, on_complete=on_complete)
        pnq.query([3, 4]).map(lambda x: x).dispatch(
            err, on_complete=on_complete
        )  # exception unhandle
        pnq.query([5, 6]).map(lambda x: x).dispatch(ok_async, on_complete=on_complete)
        pnq.query([7, 8]).map(lambda x: x).dispatch(
            err_async, on_complete=on_complete
        )  # exception unhandle
        assert succeeded == [1, 2, 5, 6]

    async def main_async():
        succeeded.clear()

        await pnq.query([1, 2]).map(lambda x: x)._.dispatch(ok, on_complete=on_complete)
        await pnq.query([3, 4]).map(lambda x: x)._.dispatch(
            err, on_complete=on_complete
        )  # exception unhandle
        await pnq.query([5, 6]).map(lambda x: x)._.dispatch(
            ok_async, on_complete=on_complete
        )
        await pnq.query([7, 8]).map(lambda x: x)._.dispatch(
            err_async, on_complete=on_complete
        )  # exception unhandle
        assert succeeded == [1, 2, 5, 6]

    main()
    asyncio.run(main_async())
