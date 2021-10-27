import asyncio
from functools import partial
from typing import AsyncIterable, TypeVar

from pnq.concurrent import get_default_pool
from pnq.inspect import is_coroutine_function
from pnq.protocols import PExecutor
from pnq.selectors import starmap

from .queries import chunked

T = TypeVar("T")


def _procceed(func, iterable):
    return [func(x) for x in iterable]


async def _procceed_async(func, iterable):
    return [await func(x) for x in iterable]


async def parallel(
    source: AsyncIterable[T], func, executor: PExecutor, *, unpack="", chunksize=1
):
    if executor is None:
        executor = get_default_pool()

    new_func = starmap(func, unpack)
    submit = executor.asubmit

    if executor.is_cpubound and chunksize != 1:
        runner = _procceed_async if asyncio.iscoroutine(func) else _procceed
        runner = partial(runner, new_func)

        tasks = [submit(runner, chunck) async for chunck in chunked(source, chunksize)]
        for task in tasks:
            for x in await task:
                yield x

    else:
        tasks = [submit(new_func, x) async for x in source]
        for task in tasks:
            yield await task


async def dispatch(
    source: AsyncIterable[T],
    func,
    executor: PExecutor = None,
    *,
    unpack="",
    chunksize=1,
    on_complete=None
):
    if executor is None:
        executor = get_default_pool()

    if on_complete is None:
        on_complete = lambda x: x  # noqa

    new_func = starmap(func, unpack)
    submit = executor.asubmit

    if executor.is_cpubound and chunksize != 1:
        runner = _procceed_async if asyncio.iscoroutine(func) else _procceed
        runner = partial(runner, new_func)

        async for chunck in chunked(source, chunksize):
            future = submit(runner, chunck)
            future.add_done_callback(on_complete)
            await asyncio.sleep(0)

    else:
        async for x in source:
            future = submit(new_func, x)
            future.add_done_callback(on_complete)
            await asyncio.sleep(0)


def exec_request(func, *args, **kwargs):
    from ..requests import Response, StopWatch

    with StopWatch() as sw:
        err = None
        result = None
        try:
            result = func(*args, **kwargs)
        except Exception as e:
            err = e

    return Response(
        func,
        args,
        kwargs,
        err=err,
        result=result,
        start=sw.start,
        end=sw.end,
    )


async def exec_request_async(func, *args, **kwargs):
    from ..requests import Response, StopWatch

    with StopWatch() as sw:
        err = None
        result = None
        try:
            result = await func(*args, **kwargs)
        except Exception as e:
            err = e

    return Response(
        func,
        args,
        kwargs,
        err=err,
        result=result,
        start=sw.start,
        end=sw.end,
    )


def request(
    source: AsyncIterable[T],
    func,
    executor: PExecutor,
    *,
    unpack="",
    chunksize=1,
    retry: int = None,
    timeout: float = None
):
    if executor is None:
        executor = get_default_pool()

    if is_coroutine_function(func):
        wrapped = partial(exec_request_async, func)
    else:
        wrapped = partial(exec_request, func)

    return parallel(source, wrapped, executor, unpack=unpack, chunksize=chunksize)
