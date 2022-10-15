import asyncio
from asyncio import iscoroutinefunction
from collections import deque
from concurrent.futures import Future
from functools import partial
from typing import Iterable, TypeVar

from pnq import selectors
from pnq.concurrent import get_default_pool
from pnq.protocols import PExecutor

from .queries import chunk

T = TypeVar("T")


def _procceed(func, iterable):
    return [func(x) for x in iterable]


async def _procceed_async(func, iterable):
    return [await func(x) for x in iterable]


def parallel(source: Iterable[T], func, executor: PExecutor, *, unpack="", chunksize=1):
    if executor is None:
        executor = get_default_pool()
    new_func = selectors.starmap(func, unpack)
    submit = executor.submit

    if executor.is_cpubound and chunksize != 1:
        runner = _procceed_async if asyncio.iscoroutine(func) else _procceed
        runner = partial(runner, new_func)

        tasks = [submit(runner, chunked) for chunked in chunk(source, chunksize)]
        for task in tasks:
            for x in task.result():
                yield x

    else:
        tasks = [submit(new_func, x) for x in source]
        for task in tasks:
            yield task.result()


def dispatch(
    source: Iterable[T],
    func,
    executor: PExecutor = None,
    *,
    unpack="",
    chunksize=1,
    on_complete=None,
):
    if executor is None:
        executor = get_default_pool()

    new_func = selectors.starmap(func, unpack)
    submit = executor.submit

    if on_complete is None:
        on_complete = lambda x: x  # noqa

    if executor.is_cpubound and chunksize != 1:
        runner = _procceed_async if asyncio.iscoroutine(func) else _procceed
        runner = partial(runner, new_func)

        for chunked in chunk(source, chunksize):
            future = submit(runner, chunked)
            future.add_done_callback(on_complete)

    else:
        for x in source:
            future = submit(new_func, x)
            future.add_done_callback(on_complete)


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
        res=result,
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
        res=result,
        start=sw.start,
        end=sw.end,
    )


def request(
    source: Iterable[T],
    func,
    executor: PExecutor,
    *,
    unpack="",
    chunksize=1,
    retry: int = None,
    timeout: float = None,
):
    if executor is None:
        executor = get_default_pool()

    if iscoroutinefunction(func):
        wrapped = partial(exec_request_async, func)
    else:
        wrapped = partial(exec_request, func)

    return parallel(source, wrapped, executor, unpack=unpack, chunksize=chunksize)


def gather(
    source: Iterable[T], parallel: int = 1, timeout=None, return_exceptions=True
):
    import time

    tasks = {}
    queue = deque()  # type: ignore

    def set_result(i, future):
        del tasks[i]
        queue.append(future)

    for i, concurrent_future in enumerate(source):
        if isinstance(concurrent_future, Future):
            tasks[i] = concurrent_future
            concurrent_future.add_done_callback(partial(set_result, i))
        else:
            yield concurrent_future.result()  # type: ignore

        while queue:
            yield queue.popleft().result()

    while tasks:
        while queue:
            yield queue.popleft().result()
        time.sleep(0.01)

    while queue:
        yield queue.popleft().result()
