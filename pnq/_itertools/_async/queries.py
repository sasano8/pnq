import asyncio
import random
from typing import AsyncIterable, TypeVar

from pnq import selectors
from pnq.exceptions import DuplicateElementError, MustError, MustTypeError

from ..common import Listable, name_as

T = TypeVar("T")


@name_as("map")
def _map(source: AsyncIterable[T], selector, unpack=""):
    if selector is None:
        return source.__aiter__()
    else:
        if selector is str:
            selector = lambda x: "" if x is None else str(x)

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


async def flat(source: AsyncIterable[T], selector=None):
    if selector is None:
        return (_ async for inner in source async for _ in inner)
    else:
        return (_ async for elm in source async for _ in selector(elm))


async def flat_recursive(source: AsyncIterable[T], selector):
    scanner = selectors.flat_recursive(selector)
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


async def chain(*iterables):

    prev = 3

    for iterable in iterables:
        type = 0
        if hasattr(iterable, "__iter__"):
            type += 1
        if hasattr(iterable, "__aiter__"):
            type += 2

        if type == 1:
            prev = 1
            may_be_async = False
        elif type == 2:
            prev = 2
            may_be_async = True
        elif type == 3:
            prev = 3
            if prev == 1:
                may_be_async = False
            elif prev == 2:
                may_be_async = True
            elif prev == 3:
                may_be_async = True
            else:
                raise TypeError(iterable, "iterable")
        else:
            raise TypeError(iterable, "iterable")

        if may_be_async:
            async for x in iterable:
                yield x
        else:
            for x in iterable:
                yield x


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
    [].join(
        [],
        lambda left, right: left.name == right.name,
        lambda left, right: (left.name, right.age),
    )

    table(User).join(Item, on=User.id == Item.id).select(User.id, Item.id)

    pass


async def group_join(source: AsyncIterable[T], size: int):
    ...


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


async def cartesian(*iterables: AsyncIterable[T]):
    # FIXME: no implemented
    import itertools

    for x in itertools.product(*iterables):  # type: ignore
        yield x


@name_as("filter")
def _filter(source: AsyncIterable[T], predicate=None, unpack=""):
    if predicate is None:
        return source.__aiter__()
    else:
        if unpack == "":
            return (x async for x in source if predicate(x))
        elif unpack == "*":
            return (x async for x in source if predicate(*x))
        elif unpack == "**":
            return (x async for x in source if predicate(**x))
        elif unpack == "***":
            return (x async for x in source if predicate(*x.args, **x.kwargs))  # type: ignore
        else:
            raise ValueError("unpack must be one of '', '*', '**', '***'")


async def must(source: AsyncIterable[T], predicate, msg=""):
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
    async for value in _map(source, selector):
        if value in duplidate:
            pass
        else:
            duplidate.add(value)
            yield value


async def must_unique(source: AsyncIterable[T], selector=None):
    duplidate = set()
    async for value in _map(source, selector):
        if value in duplidate:
            raise DuplicateElementError(value)
        else:
            duplidate.add(value)
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
    if page < 1:
        raise ValueError("page must be greater than 0")
    if size < 0:
        raise ValueError("size must be >= 0")
    start = (page - 1) * size
    stop = start + size
    return range(start, stop)


async def order_by(source: AsyncIterable[T], key_selector=None, desc: bool = False):
    for x in sorted(await Listable(source), key=key_selector, reverse=desc):
        yield x


async def min_by(source: AsyncIterable[T], key_selector=None):
    for x in sorted(await Listable(source), key=key_selector, reverse=False):
        yield x


async def max_by(source: AsyncIterable[T], key_selector=None):
    for x in sorted(await Listable(source), key=key_selector, reverse=True):
        yield x


async def order_by_reverse(source: AsyncIterable[T]):
    return reversed(await Listable(source))


async def order_by_shuffle(source: AsyncIterable[T], seed=None):
    if seed is None:
        rand = random.random
        func = lambda x: rand()  # noqa

    else:
        rand = random.Random(seed).random
        func = lambda x: rand()  # noqa

    for x in sorted(await Listable(source), key=func):
        yield x


async def sleep(source: AsyncIterable[T], seconds: float):
    sleep = asyncio.sleep
    async for v in source:
        yield v
        await sleep(seconds)


async def as_aiter(source):
    type = 0
    if hasattr(source, "__iter__"):
        type += 1
    if hasattr(source, "__aiter__"):
        type += 2

    if type == 3:
        async for x in source:
            yield x
    elif type == 2:
        async for x in source:
            yield x
    elif type == 1:
        for x in source:
            yield x
    else:
        raise TypeError(f"{source} must be iterable")
