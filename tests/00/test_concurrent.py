import asyncio

import pytest

from pnq.concurrent import AsyncPool, DummyPool, ProcessPool, ThreadPool
from tests.conftest import to_sync


def do(x):
    import time

    time.sleep(0.1)
    return x


def do_heavy(x):
    import time

    time.sleep(1)
    return x


async def do_async(x):
    await asyncio.sleep(0.1)
    return x


async def do_heavy_async(x):
    await asyncio.sleep(1)
    return x


@to_sync
async def test_base_spec():
    import pnq.concurrent.testtool as testtool

    spectest = testtool.ExecutorSpec.test_capability

    assert await spectest(
        ProcessPool, is_async_only=False, submit_sync=True, submit_async=True
    )
    assert await spectest(
        ThreadPool, is_async_only=False, submit_sync=True, submit_async=True
    )
    assert await spectest(
        AsyncPool, is_async_only=True, submit_sync=False, submit_async=False
    )
    assert await spectest(
        DummyPool, is_async_only=False, submit_sync=True, submit_async=True
    )


@to_sync
async def test_is_cpubound():
    with ProcessPool(1) as pool:
        assert pool.is_cpubound

    with ThreadPool(1) as pool:
        assert not pool.is_cpubound

    async with AsyncPool(1) as pool:
        assert not pool.is_cpubound

    async with DummyPool(1) as pool:
        assert not pool.is_cpubound


def test_sync_func():
    futures = []

    with ProcessPool(1) as pool:
        future = pool.submit(do, 1)
        futures.append(future)

    with ThreadPool(1) as pool:
        future = pool.submit(do, 2)
        futures.append(future)

    with DummyPool(1) as pool:
        future = pool.submit(do, 3)
        futures.append(future)

    assert [x.result() for x in futures] == [1, 2, 3]


def test_async_func():
    futures = []

    with ProcessPool(1) as pool:
        future = pool.submit(do_async, 1)
        futures.append(future)

    with ThreadPool(1) as pool:
        future = pool.submit(do_async, 2)
        futures.append(future)

    # with pytest.raises(TypeError, match="async function cannot be submitted"):
    with DummyPool(1) as pool:
        future = pool.submit(do_async, 3)
        futures.append(future)

    assert [x.result() for x in futures] == [1, 2, 3]


def test_sync_pool_wait_all_tasks():
    """一度スケジュールされたタスクは全て実行されるべき"""
    futures = []

    with ProcessPool(1) as pool:
        future = pool.submit(do_heavy, 1)
        futures.append(future)
        future = pool.submit(do_heavy, 2)
        futures.append(future)
        future = pool.submit(do_heavy_async, 3)
        futures.append(future)

    with ThreadPool(1) as pool:
        future = pool.submit(do_heavy, 4)
        futures.append(future)
        future = pool.submit(do_heavy, 5)
        futures.append(future)
        future = pool.submit(do_heavy_async, 6)
        futures.append(future)

    with DummyPool(1) as pool:
        future = pool.submit(do_heavy, 7)
        futures.append(future)
        future = pool.submit(do_heavy, 8)
        futures.append(future)
        future = pool.submit(do_heavy, 9)
        futures.append(future)

    assert [x.result() for x in futures] == [1, 2, 3, 4, 5, 6, 7, 8, 9]


async def to_result(futures):
    results = []
    for future in futures:
        try:
            results.append(await future)

        except BaseException as e:
            results.append(e.__class__.__name__)

    return results


@to_sync
async def test_async_immidiate():
    """一度スケジュールされたタスクは全て実行されるべき。終了時に処理予定のタスクの完了を待つべき。"""
    from time import time

    futures = []
    times = []

    pre = time()

    with ProcessPool(1) as pool:
        future = pool.asubmit(do, 1)
        futures.append(future)

    current = time()
    times.append(current - pre)
    pre = time()

    with ThreadPool(1) as pool:
        future = pool.asubmit(do, 2)
        futures.append(future)

    # async with DummyPool(1) as pool:
    #     future = pool.asubmit(do, 3)
    #     futures.append(future)

    current = time()
    times.append(current - pre)
    pre = time()

    with ThreadPool(1) as pool:
        future = pool.asubmit(do, 4)
        futures.append(future)

    current = time()
    times.append(current - pre)
    pre = time()

    async with AsyncPool(1) as pool:
        future = pool.asubmit(do, 5)
        futures.append(future)

    current = time()
    times.append(current - pre)
    pre = time()

    async with AsyncPool(1) as pool:
        future = pool.asubmit(do_async, 6)
        futures.append(future)

    current = time()
    times.append(current - pre)

    assert await to_result(futures) == [1, 2, 4, 5, 6]
    assert min(times) > 0.1


@to_sync
async def test_async_pool_size():
    """一度に実行されるタスクは、プールサイズを超えないべき"""

    futures = []
    task_counts = []

    async with AsyncPool(2) as pool:
        future = pool.asubmit(do, 1)
        futures.append(future)
        future = pool.asubmit(do, 2)
        futures.append(future)
        future = pool.asubmit(do, 3)
        futures.append(future)
        future = pool.asubmit(do, 4)
        futures.append(future)
        future = pool.asubmit(do, 5)
        futures.append(future)
        future = pool.asubmit(do, 6)
        futures.append(future)

        async def watch_max_task():
            while True:
                task_counts.append(pool.running_task_count)
                await asyncio.sleep(0.01)

        asyncio.create_task(watch_max_task())

    assert await to_result(futures) == [1, 2, 3, 4, 5, 6]
    assert max(task_counts) == 2


def iter_sync():
    yield from [1, 2, 3]


async def iter_async():
    yield 1
    yield 2
    yield 3


def mul_sync(x):
    return x * 2


async def mul_async(x):
    return x * 2


async def to_aiter(iter):
    for x in iter:
        yield x


@to_sync
async def test_context_err_handle():
    with pytest.raises(Exception, match="err"):
        with ProcessPool(2) as pool:
            raise Exception("err")

    with pytest.raises(Exception, match="err"):
        with ThreadPool(2) as pool:
            raise Exception("err")

    with pytest.raises(NotImplementedError):
        with AsyncPool(2) as pool:
            ...

    with pytest.raises(Exception, match="err"):
        async with ProcessPool(2) as pool:
            raise Exception("err")

    with pytest.raises(Exception, match="err"):
        async with ThreadPool(2) as pool:
            raise Exception("err")

    with pytest.raises(Exception, match="err"):
        async with AsyncPool(2) as pool:
            raise Exception("err")


@to_sync
async def test_capability():
    """
    1. submitが実行可能（concurrent.Future）なエクゼキュータは同期実行できるべき。
    2. 非同期イテレータがソースの場合は、非同期でしか実行できないべき
    3. asubmitが実行可能（asyncio.Future）なエクゼキュータは非同期実行できるべき。
    4. submit/asubmitはコルーチン関数を受け入れ可能であるべき
    """

    with ProcessPool(2) as pool:
        assert list(pool.map(iter_sync(), mul_sync)) == [2, 4, 6]
        assert list(pool.map(iter_sync(), mul_async)) == [2, 4, 6]

        with pytest.raises(TypeError, match="object is not iterable"):
            list(pool.map(iter_async(), mul_sync))

        with pytest.raises(TypeError, match="object is not iterable"):
            list(pool.map(iter_async(), mul_async))

        assert await pool.map(iter_sync(), mul_sync) == [2, 4, 6]
        assert await pool.map(iter_sync(), mul_async) == [2, 4, 6]
        assert await pool.map(iter_async(), mul_sync) == [2, 4, 6]
        assert await pool.map(iter_async(), mul_async) == [2, 4, 6]

    with ThreadPool(2) as pool:
        assert list(pool.map(iter_sync(), mul_sync)) == [2, 4, 6]
        assert list(pool.map(iter_sync(), mul_async)) == [2, 4, 6]

        with pytest.raises(TypeError, match="object is not iterable"):
            list(pool.map(iter_async(), mul_sync))

        with pytest.raises(TypeError, match="object is not iterable"):
            list(pool.map(iter_async(), mul_async))

        assert await pool.map(iter_sync(), mul_sync) == [2, 4, 6]
        assert await pool.map(iter_sync(), mul_async) == [2, 4, 6]
        assert await pool.map(iter_async(), mul_sync) == [2, 4, 6]
        assert await pool.map(iter_async(), mul_async) == [2, 4, 6]

    async with AsyncPool(2) as pool:
        with pytest.raises(TypeError, match="are only allowed async"):
            list(pool.map(iter_sync(), mul_sync))

        with pytest.raises(TypeError, match="are only allowed async"):
            list(pool.map(iter_sync(), mul_async))

        with pytest.raises(TypeError, match="are only allowed async"):
            list(pool.map(iter_async(), mul_sync))

        with pytest.raises(TypeError, match="are only allowed async"):
            list(pool.map(iter_async(), mul_async))

        assert await pool.map(iter_sync(), mul_sync) == [2, 4, 6]
        assert await pool.map(iter_sync(), mul_async) == [2, 4, 6]
        assert await pool.map(iter_async(), mul_sync) == [2, 4, 6]
        assert await pool.map(iter_async(), mul_async) == [2, 4, 6]
