import asyncio
from asyncio import futures
from functools import partial
from typing import AsyncIterable, TypeVar

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
    executor: PExecutor,
    *,
    unpack="",
    chunksize=1,
    callback=None
):
    new_func = starmap(func, unpack)
    submit = executor.asubmit

    if callback:
        cb = lambda x: callback(x)
    else:
        cb = lambda x: x

    if executor.is_cpubound and chunksize != 1:
        runner = _procceed_async if asyncio.iscoroutine(func) else _procceed
        runner = partial(runner, new_func)

        async for chunck in chunked(source, chunksize):
            future = submit(runner, chunck)
            future.add_done_callback(cb)
            await asyncio.sleep(0)

    else:
        async for x in source:
            future = submit(new_func, x)
            future.add_done_callback(cb)
            await asyncio.sleep(0)
