from typing import List

import pnq
from pnq import concurrent
from pnq._itertools.requests import Response
from tests.conftest import to_sync


def mul(x):
    return x * 2


async def mul_async(x):
    return x * 2


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

    def err(value1, value2):
        raise Exception("error")

    async def ok_async(value):
        result.append(value)
        return "ok"

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
        assert res.kwargs == {"value1": 1, "value2": 2}
        assert str(res.err) == "error"
        assert res.result is None
        assert isinstance(res.start, datetime)
        assert isinstance(res.end, datetime)

        return True

    assert main(ok, err)
    assert main(ok_async, err_async)
