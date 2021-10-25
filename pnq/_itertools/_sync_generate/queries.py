import asyncio
from concurrent.futures import Future as ConcurrentFuture
from functools import partial
from typing import Iterable, TypeVar

from ...selectors import flat_recursive as _flat_recursive
from ...selectors import map as unpacking
from ..common import Listable, name_as
from ..exceptions import DuplicateElementError, MustError, MustTypeError
from ..protocols import PExecutor

T = TypeVar("T")


@name_as("map")
def _map(source: Iterable[T], selector, unpack=""):
    if selector is None:
        return source.__iter__()
    else:
        if selector is str:
            selector = lambda x: "" if x is None else str(x)

        if unpack == "":
            return (selector(x) for x in source)
        elif unpack == "*":
            return (selector(*x) for x in source)
        elif unpack == "**":
            return (selector(**x) for x in source)
        elif unpack == "***":
            return (selector(*x.args, **x.kwargs) for x in source)  # type: ignore
        else:
            raise ValueError("unpack must be one of '', '*', '**', '***'")


def gather(source: Iterable[T], selector=None, parallel: int = 1, timeout=None):
    if selector is None:
        selector = lambda x: x  # noqa

    for tag, result in gather_tagged(
        _enumerate(Listable(source, selector)), parallel=parallel, timeout=timeout
    ):
        yield result


def gather_tagged(source: Iterable[T], selector=None, parallel: int = 1, timeout=None):
    if selector is None:
        selector = lambda x: x  # noqa

    if parallel > 1:
        for chunk in chunked(source, size=parallel):
            results = asyncio.gather(
                *(asyncio.wait_for(x, timeout) for tag, x in Listable(chunk, selector))
            )
            for tagged_result in ((chunk[i][0], x) for i, x in enumerate(results)):
                yield tagged_result
    else:
        for x in source:
            tag, awaitable = selector(x)
            yield tag, asyncio.wait_for(awaitable, timeout)


def schedule(pnq_future_coro):
    if asyncio.iscoroutine(pnq_future_coro):
        return asyncio.create_task(pnq_future_coro)
    elif isinstance(pnq_future_coro, ConcurrentFuture):
        return asyncio.wrap_future(pnq_future_coro)
    else:
        raise NotImplementedError()


def flat(source: Iterable[T], selector=None):
    if selector is None:
        return (_ for inner in source for _ in inner)
    else:
        return (_ for elm in source for _ in selector(elm))


def flat_recursive(source: Iterable[T], selector):
    scanner = _flat_recursive(selector)
    for node in source:
        for x in scanner(node):
            yield x


def pivot_unstack(source: Iterable[T], default=None):
    dataframe = {}  # type: ignore
    data = []

    # 全てのカラムを取得
    for i, dic in _enumerate(source):
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


def pivot_stack(source: Iterable[T], default=None):
    data = dict(Listable(source))
    columns = list(data.keys())

    for i in range(len(columns)):
        row = {}
        for c in columns:
            row[c] = data[c][i]

        yield row


@name_as("enumerate")
def _enumerate(source: Iterable[T], start: int = 0, step: int = 1):
    i = start
    for x in source:
        yield i, x
        i += step


def group_by(source: Iterable[T], selector):
    from collections import defaultdict

    results = defaultdict(list)
    for elm in source:
        k, v = selector(elm)
        results[k].append(v)

    for k, v in results.items():
        yield k, v


def chunked(source: Iterable[T], size: int):
    current = 0
    running = True

    it = source.__iter__()

    while running:
        current = 0
        queue = []

        while current < size:
            current += 1
            try:
                val = it.__next__()
                queue.append(val)
            except StopIteration:
                running = False
                break

        if queue:
            yield queue


def tee(source: Iterable[T], size: int):
    ...


def join(source: Iterable[T], size: int):
    ...


def group_join(source: Iterable[T], size: int):
    ...


def request(source: Iterable[T], func, retry: int = None):
    from ..requests import Response, StopWatch

    for v in source:

        with StopWatch() as sw:
            err = None
            result = None
            try:
                result = func(**v)
            except Exception as e:
                err = e

        res = Response(
            func, kwargs=v, err=err, result=result, start=sw.start, end=sw.end
        )

        yield res


def request_async(source: Iterable[T], func, timeout: float = None, retry: int = None):
    from ..requests import Response, StopWatch

    for v in source:

        with StopWatch() as sw:
            err = None
            result = None
            try:
                result = func(**v)
            except Exception as e:
                err = e

        res = Response(
            func, kwargs=v, err=err, result=result, start=sw.start, end=sw.end
        )

        yield res


def _procceed(func, iterable):
    return [func(x) for x in iterable]


def _procceed_async(func, iterable):
    return [func(x) for x in iterable]


def parallel(source: Iterable[T], func, executor: PExecutor, *, unpack="", chunksize=1):
    new_func = unpacking(func, unpack)
    submit = executor.asubmit

    if executor.is_cpubound and chunksize != 1:
        runner = _procceed_async if asyncio.iscoroutine(func) else _procceed
        runner = partial(runner, new_func)

        tasks = [submit(runner, chunck) for chunck in chunked(source, chunksize)]
        for task in tasks:
            for x in task:
                yield x

    else:
        tasks = [submit(new_func, x) for x in source]
        for task in tasks:
            yield task


def debug(source: Iterable[T], breakpoint=lambda x: x, printer=print):
    for v in source:
        printer(v)
        breakpoint(v)
        yield v


def union_all(source: Iterable[T]):
    ...


extend = union_all


def union(source: Iterable[T]):
    ...


def union_intersect(source: Iterable[T]):
    ...


def union_minus(source: Iterable[T]):
    ...


# difference
# symmetric_difference

# < <= > >= inclusion


@name_as("zip")
def _zip(source: Iterable[T]):
    ...


def compress(source: Iterable[T]):
    import itertools

    return itertools.compress(self, *iterables)


def cartesian(*iterables: Iterable[T]):
    # FIXME: no implemented
    import itertools

    for x in itertools.product(*iterables):  # type: ignore
        yield x


@name_as("filter")
def _filter(source: Iterable[T], predicate=None, unpack=""):
    if predicate is None:
        return source.__iter__()
    else:
        if unpack == "":
            return (x for x in source if predicate(x))
        elif unpack == "*":
            return (x for x in source if predicate(*x))
        elif unpack == "**":
            return (x for x in source if predicate(**x))
        elif unpack == "***":
            return (x for x in source if predicate(*x.args, **x.kwargs))  # type: ignore
        else:
            raise ValueError("unpack must be one of '', '*', '**', '***'")


def must(source: Iterable[T], predicate, msg=""):
    for elm in source:
        if not predicate(elm):
            raise MustError(f"{msg} {elm}")
        yield elm


def filter_type(source: Iterable[T], *types):
    for elm in source:
        if isinstance(elm, *types):
            yield elm


def must_type(source: Iterable[T], *types):
    for elm in source:
        if not isinstance(elm, types):
            raise MustTypeError(f"{elm} is not {types}")
        yield elm


def filter_keys(source: Iterable[T], *keys):
    pass


def must_keys(source: Iterable[T], *keys):
    pass


def filter_unique(source: Iterable[T], selector=None):
    duplidate = set()
    for value in _map(source, selector):
        if value in duplidate:
            pass
        else:
            duplidate.add(value)
            yield value


def must_unique(source: Iterable[T], selector=None):
    duplidate = set()
    for value in _map(source, selector):
        if value in duplidate:
            raise DuplicateElementError(value)
        else:
            duplidate.add(value)
            yield value


def take(source: Iterable[T], range: range):
    start = range.start
    stop = range.stop
    step = range.step

    current = 0

    it = source.__iter__()

    try:
        while current < start:
            it.__next__()
            current += step
    except StopIteration:
        return

    try:
        while current < stop:
            yield it.__next__()
            current += step
    except StopIteration:
        return


def take_while(source: Iterable[T], predicate):
    for v in source:
        if predicate(v):
            yield v
        else:
            break


def skip(source: Iterable[T], range: range):
    start = range.start
    stop = range.stop

    current = 0

    it = source.__iter__()

    try:
        while current < start:
            yield it.__next__()
            current += 1
    except StopIteration:
        return

    try:
        while current < stop:
            it.__next__()
            current += 1
    except StopIteration:
        return

    for x in it:
        yield x


def skip_while(source: Iterable[T], predicate):
    for v in source:
        if not predicate(v):
            break

    for v in source:
        yield v


def take_page(source: Iterable[T], page: int, size: int):
    r = _take_page_calc(page, size)
    return take(source, r)


def _take_page_calc(page: int, size: int):
    if size < 0:
        raise ValueError("size must be >= 0")
    start = (page - 1) * size
    stop = start + size
    return range(start, stop)


def order_by(source: Iterable[T], selector=None, desc: bool = False):
    for x in sorted(Listable(source), key=selector, reverse=desc):
        yield x


def order_by_reverse(source: Iterable[T]):
    return reversed(Listable(source))


def order_by_shuffle(source: Iterable[T]):
    import random

    for x in sorted(Listable(source), key=lambda k: random.random()):
        yield x


def sleep(source: Iterable[T], seconds: float):
    sleep = asyncio.sleep
    for v in source:
        yield v
        sleep(seconds)
