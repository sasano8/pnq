import pnq
from pnq import concurrent
from pnq._itertools.requests import Response
from tests.conftest import to_sync


def mul(x):
    return x * 2


async def mul_async(x):
    return x * 2


@to_sync
async def test_concurrent():
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
