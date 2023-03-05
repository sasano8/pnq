import asyncio
import itertools
import time
from concurrent.futures import Future as ConcurrentFuture
from functools import partial
from typing import Iterable, Iterator, Literal

from .iterables import LazyListable


async def list_from_aiter(aiter):
    return [x async for x in aiter]


def map_iter(iterable, func, unpack: Literal["", "*", "**", "***"] = ""):
    if unpack == "":
        return (func(x) for x in iterable)
    elif unpack == "*":
        return (func(*x) for x in iterable)
    elif unpack == "**":
        return (func(**x) for x in iterable)
    elif unpack == "***":
        return (func(*x.args, **x.kwargs) for x in iterable)
    else:
        raise ValueError()


def map_iter_submit(
    executor, iterable, func, unpack: Literal["", "*", "**", "***"] = ""
) -> Iterator[ConcurrentFuture]:
    if unpack == "":
        return (executor.submit(func, x) for x in iterable)
    elif unpack == "*":
        return (executor.submit(func, *x) for x in iterable)
    elif unpack == "**":
        return (executor.submit(func, **x) for x in iterable)
    elif unpack == "***":
        return (executor.submit(func, *x.args, **x.kwargs) for x in iterable)
    else:
        raise ValueError()


def map_iter_asubmit(
    executor, iterable, func, unpack: Literal["", "*", "**", "***"] = ""
) -> Iterable[asyncio.Future]:
    if unpack == "":
        return (executor.asubmit(func, x) for x in iterable)
    elif unpack == "*":
        return (executor.asubmit(func, *x) for x in iterable)
    elif unpack == "**":
        return (executor.asubmit(func, **x) for x in iterable)
    elif unpack == "***":
        return (executor.asubmit(func, *x.args, **x.kwargs) for x in iterable)
    else:
        raise ValueError()


async def map_to_aiter(iterable, func, unpack: Literal["", "*", "**", "***"] = ""):
    if unpack == "":
        for x in iterable:
            yield await func(x)
    elif unpack == "*":
        for x in iterable:
            yield await func(*x)
    elif unpack == "**":
        for x in iterable:
            yield await func(**x)
    elif unpack == "***":
        for x in iterable:
            yield await func(*x.args, **x.kwargs)
    else:
        raise ValueError()


async def map_aiter(iterable, func, unpack: Literal["", "*", "**", "***"] = ""):
    if asyncio.iscoroutinefunction(func):
        if unpack == "":
            async for x in iterable:
                yield await func(x)
        elif unpack == "*":
            async for x in iterable:
                yield await func(*x)
        elif unpack == "**":
            async for x in iterable:
                yield await func(**x)
        elif unpack == "***":
            async for x in iterable:
                yield await func(*x.args, **x.kwargs)
        else:
            raise ValueError()
    else:
        if unpack == "":
            async for x in iterable:
                yield func(x)
        elif unpack == "*":
            async for x in iterable:
                yield func(*x)
        elif unpack == "**":
            async for x in iterable:
                yield func(**x)
        elif unpack == "***":
            async for x in iterable:
                yield func(*x.args, **x.kwargs)
        else:
            raise ValueError()


def process_chunk(func, chunk, unpack: Literal["", "*", "**"] = ""):
    return list(map_iter(chunk, func, unpack=unpack))


def process_chunk_async(
    func, chunk, unpack: Literal["", "*", "**"] = "", async_runner=None
):
    return async_runner(list_from_aiter(map_to_aiter(chunk, func, unpack=unpack)))


def take(iter, *args):
    return itertools.islice(iter, *args)


async def take_async(aiter, *args):
    r = range(*args)
    start = r.start
    stop = r.stop
    step = r.step

    current = 0
    it = aiter.__aiter__()

    try:
        while current < start:
            await it.__anext__()
            current += 1
    except StopAsyncIteration:
        return

    try:
        while current < stop:
            result = await it.__anext__()
            current += 1
            if (current % step) == 0:
                yield result

    except StopAsyncIteration:
        return


def chunk(iterable, chunksize: int):
    it = iter(iterable)
    while True:
        chunked = list(take(it, chunksize))
        if not chunked:
            return
        yield chunked


async def chunk_async(aiter, chunksize: int):
    it = aiter.__aiter__()
    while True:
        chunked = [x async for x in take_async(it, chunksize)]
        if not chunked:
            return
        yield chunked


def chain_chunks(iterable):
    for element in iterable:
        element.reverse()
        while element:
            yield element.pop()


def map_each_chunk(
    executor,
    iterable,
    func,
    unpack: Literal["", "*", "**"] = "",
    timeout=None,
    chunksize=1,
):
    if chunksize < 1:
        raise ValueError("chunksize must be >= 1.")

    if asyncio.iscoroutinefunction(func):
        processer = process_chunk_async
    else:
        processer = process_chunk

    results = map(
        executor,
        chunk(iterable, chunksize=chunksize),
        partial(processer, func, unpack=unpack),
        timeout=timeout,
    )
    return chain_chunks(results)


def map(
    executor,
    iterable,
    func,
    unpack: Literal["", "*", "**"] = "",
    timeout=None,
):
    return LazyListable(Map(executor, iterable, func, unpack, timeout))


class IterBase:
    def __iter__(self):
        return self._iter_impl()

    def __aiter__(self):
        return self._aiter_impl()

    @staticmethod
    def get_iter_type(iterable):
        if hasattr(iterable, "__aiter__"):
            if hasattr(iterable, "__iter__"):
                return 3
            else:
                return 2
        else:
            if hasattr(iterable, "__iter__"):
                return 1
            else:
                return 0

    @classmethod
    def _get_is_aiter(cls, iterable):
        type = cls.get_iter_type(iterable)
        if type == 2:
            return True
        elif type == 1:
            return False
        elif type == 3:
            raise TypeError(
                f"Could not identify {iterable} as '__iter__' or '__aiter__'"
            )
        else:
            raise TypeError(f"{iterable} has no '__iter__' or '__aiter__'")

    def _iter_impl(self):
        raise NotImplementedError()

    def _aiter_impl(self):
        raise NotImplementedError()


class Chunk(IterBase):
    def __init__(
        self,
        iterable,
        chunksize=1,
    ):
        self.source = iterable
        self.chunksize = chunksize

    def _iter_impl(self):
        it = iter(self.source)
        while True:
            chunked = list(itertools.islice(it, self.chunksize))
            if not chunked:
                return
            yield chunked

    async def _aiter_impl(self):
        it = self.source.__aiter__()
        while True:
            chunked = [x async for x in take_async(it, self.chunksize)]
            if not chunked:
                return
            yield chunked


# TODO: 塊ごとに１ワーカーで処理するのか、要素ごとに１ワーカーで処理するのか使い分けれるといい
class Map(IterBase):
    def __init__(
        self,
        executor,
        iterable,
        func,
        unpack: Literal["", "*", "**"] = "",
        timeout=None,
        chunksize=None,
    ):
        if chunksize is None:
            chunksize = executor.max_workers

        self._get_is_aiter(iterable)

        self.executor = executor
        self.source = iterable
        self.func = func
        self.unpack = unpack
        self.timeout = timeout

    def __iter__(self):
        if getattr(self.executor, "is_async_only", False):
            raise TypeError(
                f"{self.executor} are only allowed async. use async for or await."
            )
        return self._iter_impl()

    def _iter_impl(self):
        return self._get_iter(
            self.executor,
            self.source,
            self.func,
            self.unpack,
            self.timeout,
        )

    def _aiter_impl(self):
        return self._get_aiter(
            self.executor,
            self.source,
            self.func,
            self.unpack,
            self.timeout,
        )

    @classmethod
    def _get_iter(cls, executor, it, func, unpack, timeout, chunksize=1):
        if timeout is not None:
            end_time = timeout + time.monotonic()

        def result_iterator(chunk: list):
            try:
                # reverse to keep finishing order
                fs.reverse()
                while fs:
                    # Careful not to keep a reference to the popped future
                    if timeout is None:
                        yield fs.pop().result()
                    else:
                        yield fs.pop().result(end_time - time.monotonic())
            finally:
                for future in fs:
                    future.cancel()

        for chunked in Chunk(it, chunksize):
            fs = list(map_iter_submit(executor, chunked, func, unpack=unpack))
            for x in result_iterator(fs):
                yield x

    @classmethod
    async def _get_aiter(cls, executor, it, func, unpack, timeout, chunksize=1):
        if timeout is not None:
            end_time = timeout + time.monotonic()

        async def result_iterator(chunk: list):
            try:
                # reverse to keep finishing order
                fs.reverse()
                while fs:
                    # Careful not to keep a reference to the popped future
                    if timeout is None:
                        yield await fs.pop()
                    else:
                        yield await asyncio.wait_for(
                            fs.pop(), timeout=end_time - time.monotonic()
                        )
            finally:
                for future in fs:
                    future.cancel()

        is_async = cls._get_is_aiter(it)

        if is_async:
            async for chunked in Chunk(it, chunksize):
                fs = list(map_iter_asubmit(executor, chunked, func, unpack=unpack))
                async for x in result_iterator(fs):
                    yield x

        else:
            for chunked in Chunk(it, chunksize):
                fs = list(map_iter_asubmit(executor, chunked, func, unpack=unpack))
                async for x in result_iterator(fs):
                    yield x
