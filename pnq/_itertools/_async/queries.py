import asyncio
from concurrent.futures import Future as ConcurrentFuture
from functools import partial
from typing import AsyncIterable, TypeVar

from ...selectors import flat_recursive as _flat_recursive
from ...selectors import map as unpacking
from ..common import Listable, name_as
from ..protocols import PExecutor

T = TypeVar("T")


@name_as("map")
def _map(source: AsyncIterable[T], selector, unpack=""):
    if selector is None:
        return source
    else:
        if unpack == "":
            return (selector(x) async for x in source)
        elif unpack == "*":
            return (selector(*x) async for x in source)
        elif unpack == "**":
            return (selector(**x) async for x in source)
        elif unpack == "***":
            return (selector(*x.args, **x.kwargs) async for x in source)  # type: ignore
        else:
            raise ValueError("unpack must be one of '', '*', '**', '***'")


async def gather(
    source: AsyncIterable[T], selector=None, parallel: int = 1, timeout=None
):
    if selector is None:
        selector = lambda x: x  # noqa

    async for tag, result in gather_tagged(
        _enumerate(Listable(source, selector)), parallel=parallel, timeout=timeout
    ):
        yield result


async def gather_tagged(
    source: AsyncIterable[T], selector=None, parallel: int = 1, timeout=None
):
    if selector is None:
        selector = lambda x: x  # noqa

    if parallel > 1:
        async for chunk in chunked(source, size=parallel):
            results = await asyncio.gather(
                *(asyncio.wait_for(x, timeout) for tag, x in Listable(chunk, selector))
            )
            for tagged_result in ((chunk[i][0], x) for i, x in enumerate(results)):
                yield tagged_result
    else:
        async for x in source:
            tag, awaitable = selector(x)
            yield tag, await asyncio.wait_for(awaitable, timeout)


def schedule(pnq_future_coro):
    if asyncio.iscoroutine(pnq_future_coro):
        return asyncio.create_task(pnq_future_coro)
    elif isinstance(pnq_future_coro, ConcurrentFuture):
        return asyncio.wrap_future(pnq_future_coro)
    else:
        raise NotImplementedError()


async def flat(source: AsyncIterable[T], selector=None):
    if selector is None:
        return (_ async for inner in source async for _ in inner)
    else:
        return (_ async for elm in source async for _ in selector(elm))


async def flat_recursive(source: AsyncIterable[T], selector):
    scanner = _flat_recursive(selector)
    async for node in source:
        for x in scanner(node):
            yield x


async def pivot_unstack(source: AsyncIterable[T], default=None):
    dataframe = {}  # type: ignore
    data = []

    # 全てのカラムを取得
    async for i, dic in _enumerate(source):
        data.append(dic)
        for k in dic.keys():
            dataframe[k] = None

    # カラム分の領域を初期化
    for k in dataframe:
        dataframe[k] = []

    # データをデータフレームに収める
    for dic in data:
        for k in dataframe.keys():
            v = dic.get(k, default)
            dataframe[k].append(v)

    for x in dataframe.items():
        yield x


async def pivot_stack(source: AsyncIterable[T], default=None):
    data = dict(await Listable(source))
    columns = list(data.keys())

    for i in range(len(columns)):
        row = {}
        for c in columns:
            row[c] = data[c][i]

        yield row


@name_as("enumerate")
async def _enumerate(source: AsyncIterable[T], start: int = 0, step: int = 1):
    i = start
    async for x in source:
        yield i, x
        i += step


async def group_by(source: AsyncIterable[T], selector):
    from collections import defaultdict

    results = defaultdict(list)
    async for elm in source:
        k, v = selector(elm)
        results[k].append(v)

    for k, v in results.items():
        yield k, v


async def chunked(source: AsyncIterable[T], size: int):
    current = 0
    running = True

    it = source.__aiter__()

    while running:
        current = 0
        queue = []

        while current < size:
            current += 1
            try:
                val = await it.__anext__()
                queue.append(val)
            except StopAsyncIteration:
                running = False
                break

        if queue:
            yield queue


async def tee(source: AsyncIterable[T], size: int):
    ...


async def join(source: AsyncIterable[T], size: int):
    ...


async def group_join(source: AsyncIterable[T], size: int):
    ...


async def request(source: AsyncIterable[T], func, retry: int = None):
    ...


async def request_async(source: AsyncIterable[T], func, retry: int = None):
    ...


def _procceed(func, iterable):
    return [func(x) for x in iterable]


async def _procceed_async(func, iterable):
    return [await func(x) for x in iterable]


async def parallel(
    source: AsyncIterable[T], func, executor: PExecutor, *, unpack="", chunksize=1
):
    new_func = unpacking(func, unpack)
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


async def debug(source: AsyncIterable[T], breakpoint=lambda x: x, printer=print):
    async for v in source:
        printer(v)
        breakpoint(v)
        yield v


async def union_all(source: AsyncIterable[T]):
    ...


extend = union_all


async def union(source: AsyncIterable[T]):
    ...


async def union_intersect(source: AsyncIterable[T]):
    ...


async def union_minus(source: AsyncIterable[T]):
    ...


# difference
# symmetric_difference

# < <= > >= inclusion


@name_as("zip")
async def _zip(source: AsyncIterable[T]):
    ...


async def compress(source: AsyncIterable[T]):
    import itertools

    return itertools.compress(self, *iterables)


async def cartesian(source: AsyncIterable[T], *iterables):
    return itertools.product(self.source, *self.iterables)


@name_as("filter")
def _filter(source: AsyncIterable[T], predicate=None, unpack=""):
    if predicate is None:
        return source
    else:
        if unpack == "":
            return (predicate(x) async for x in source)
        elif unpack == "*":
            return (predicate(*x) async for x in source)
        elif unpack == "**":
            return (predicate(**x) async for x in source)
        elif unpack == "***":
            return (predicate(*x.args, **x.kwargs) async for x in source)  # type: ignore
        else:
            raise ValueError("unpack must be one of '', '*', '**', '***'")


async def must(source: AsyncIterable[T], predicate):
    async for elm in source:
        if not predicate(elm):
            raise MustError(f"{msg} {elm}")
        yield elm


async def filter_type(source: AsyncIterable[T], *types):
    async for elm in source:
        if isinstance(elm, *types):
            yield elm


async def must_type(source: AsyncIterable[T], *types):
    async for elm in source:
        if not isinstance(elm, types):
            raise MustTypeError(f"{elm} is not {types}")
        yield elm


async def filter_keys(source: AsyncIterable[T], *keys):
    pass


async def must_keys(source: AsyncIterable[T], *keys):
    pass


async def filter_unique(source: AsyncIterable[T], selector=None):
    duplidate = set()
    async for value in source:
        if value in duplidate:
            pass
        else:
            duplidate.add(value)
            yield value


async def must_unique(source: AsyncIterable[T], selector=None):
    duplidate = set()
    async for value in source:
        target = selector(value)
        if target in duplidate:
            raise DuplicateElementError(value)
        else:
            duplidate.add(target)
            yield value


async def take(source: AsyncIterable[T], range: range):
    start = range.start
    stop = range.stop
    step = range.step

    current = 0

    it = source.__aiter__()

    try:
        while current < start:
            await it.__anext__()
            current += step
    except StopAsyncIteration:
        return

    try:
        while current < stop:
            yield await it.__anext__()
            current += step
    except StopAsyncIteration:
        return


async def take_while(source: AsyncIterable[T], predicate):
    async for v in source:
        if predicate(v):
            yield v
        else:
            break


async def skip(source: AsyncIterable[T], range: range):
    start = range.start
    stop = range.stop

    current = 0

    it = source.__aiter__()

    try:
        while current < start:
            yield await it.__anext__()
            current += 1
    except StopAsyncIteration:
        return

    try:
        while current < stop:
            await it.__anext__()
            current += 1
    except StopAsyncIteration:
        return

    async for x in it:
        yield x


async def skip_while(source: AsyncIterable[T], predicate):
    async for v in source:
        if not predicate(v):
            break

    async for v in source:
        yield v


def take_page(source: AsyncIterable[T], page: int, size: int):
    r = _take_page_calc(page, size)
    return take(source, r)


def _take_page_calc(page: int, size: int):
    if size < 0:
        raise ValueError("size must be >= 0")
    start = (page - 1) * size
    stop = start + size
    return range(start, stop)


async def order_by(source: AsyncIterable[T], selector=None, desc: bool = False):
    for x in sorted(await Listable(source), key=selector, reverse=desc):
        yield x


async def order_by_reverse(source: AsyncIterable[T]):
    return reversed(await Listable(source))


async def order_by_shuffle(source: AsyncIterable[T]):
    import random

    for x in sorted(await Listable(source), key=lambda k: random.random()):
        yield x
