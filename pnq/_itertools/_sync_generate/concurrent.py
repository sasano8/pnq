import asyncio
from collections import deque
from functools import partial
from typing import Iterable, TypeVar

from pnq import selectors
from pnq.concurrent import get_default_pool
from pnq.inspect import is_coroutine_function
from pnq.protocols import PExecutor
from pnq.selectors import starmap

from ..common import Listable
from .queries import _enumerate, chunk

T = TypeVar("T")


def _procceed(func, iterable):
    return [func(x) for x in iterable]


def _procceed_async(func, iterable):
    return [func(x) for x in iterable]


def parallel(source: Iterable[T], func, executor: PExecutor, *, unpack="", chunksize=1):
    if executor is None:
        executor = get_default_pool()

    new_func = starmap(func, unpack)
    submit = executor.asubmit

    if executor.is_cpubound and chunksize != 1:
        runner = _procceed_async if asyncio.iscoroutine(func) else _procceed
        runner = partial(runner, new_func)

        tasks = [submit(runner, chuncked) for chuncked in chunk(source, chunksize)]
        for task in tasks:
            for x in task:
                yield x

    else:
        tasks = [submit(new_func, x) for x in source]
        for task in tasks:
            yield task


def dispatch(
    source: Iterable[T],
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

        for chuncked in chunk(source, chunksize):
            future = submit(runner, chuncked)
            future.add_done_callback(on_complete)
            asyncio.sleep(0)

    else:
        for x in source:
            future = submit(new_func, x)
            future.add_done_callback(on_complete)
            asyncio.sleep(0)


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


def exec_request_async(func, *args, **kwargs):
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


def request(
    source: Iterable[T],
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


def gather(
    source: Iterable[T], parallel: int = 1, timeout=None, return_exceptions=True
):
    tasks = {}
    results = deque()

    def set_result(i, future):
        del tasks[i]
        results.append(future)

    for i, x in _enumerate(Listable(source, selectors._select_future_with_schedule)):
        x.add_done_callback(partial(set_result, i))
        tasks[i] = x
        while results:
            yield results.popleft().result()

    while tasks:
        while results:
            yield results.popleft().result()
        asyncio.sleep(0.01)

    while results:
        yield results.popleft().result()


def call_func(sem, x, timeout):
    with sem:
        return asyncio.wait_for(x, timeout)


def gather_tagged(source: Iterable[T], selector=None, parallel: int = 1, timeout=None):
    if parallel > 1:
        tasks = []
        sem = asyncio.Semaphore(parallel)
        for tag, x in Listable(source, selector):
            task = asyncio.create_task(call_func(sem, x, timeout))
            tasks.append((tag, task))
            asyncio.sleep(0)

            if tasks[0][1].done():
                tag, task = tasks.pop(0)
                yield tag, task.result()

        while tasks:
            tag, task = tasks.pop(0)
            yield tag, task

    else:
        for tag, awaitable in Listable(source, selector):
            yield tag, asyncio.wait_for(awaitable, timeout)
