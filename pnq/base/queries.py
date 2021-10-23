import asyncio
import itertools
from collections import defaultdict
from operator import attrgetter, itemgetter
from typing import Awaitable, Callable, Mapping, NoReturn, TypeVar, Union

from .. import selectors as _selectors
from .core import IterType, Query, QueryDict, QuerySeq, QuerySet
from .exceptions import (
    DuplicateElementError,
    MustError,
    MustTypeError,
    NoElementError,
    NotFoundError,
    NotOneElementError,
)

R = TypeVar("R")

exports = []


def mark(cls):
    exports.append(cls)
    return cls


@mark
class Lazy(Query, Awaitable[R]):
    def __init__(self, source, finalizer: Callable[..., R], *args, **kwargs):
        super().__init__(source)
        self.finalizer = finalizer
        self.args = args
        self.kwargs = kwargs

    def __call__(self) -> R:
        return self.finalizer(self.__iter__(), *self.args, **self.kwargs)

    def __await__(self):
        coro = self.finalizer(self.__aiter__(), *self.args, **self.kwargs)
        return coro.__await__()


@mark
class AsyncMap(Query):
    iter_type = IterType.ASYNC

    def __init__(self, source, selector):
        super().__init__(source)
        self.selector = selector

    async def _impl_aiter(self):
        selector = self.selector
        async for x in self.source:
            yield await selector(x)


@mark
class Sleep(Query):
    iter_type = IterType.ASYNC

    def __init__(self, source, seconds):
        super().__init__(source)
        self.seconds = seconds

    async def _impl_aiter(self):
        seconds = self.seconds
        sleep = asyncio.sleep

        async for v in self.source:
            yield v
            await sleep(seconds)


@mark
class MapNullable(Query):
    def __init__(self, source, selector=None):
        Query.__init__(self, source)
        if selector is None:
            self.selector = selector
        else:
            self.selector = _selectors.map(selector)

    def _impl_iter(self):
        selector = self.selector
        if selector is None:
            return self.source.__iter__()
        else:
            return map(selector, self.source)

    def _impl_aiter(self):
        selector = self.selector
        if selector is None:
            return self.source.__aiter__()
        else:
            return (selector(x) async for x in self.source)


@mark
class Map(MapNullable):
    def __init__(self, source, selector):
        Query.__init__(self, source)
        self.selector = _selectors.map(selector)


@mark
class MapAwait(Query):
    def __init__(self, source):
        super().__init__(source)

    def _impl_iter(self):
        as_future = inspect.as_future
        for futurable in self.source:
            future = as_future(futurable)
            yield future.result()

    async def _impl_aiter(self):
        as_future = inspect.as_afuture
        async for futurable in self.source:
            future = as_future(futurable)
            yield await future


@mark
class UnpackPos(Map):
    def __init__(self, source, selector):
        selector = _selectors.map(selector, unpack="*")
        super().__init__(source, selector)


@mark
class UnpackKw(Map):
    def __init__(self, source, selector):
        selector = _selectors.map(selector, unpack="**")
        super().__init__(source, selector)


@mark
class Select(Map):
    def __init__(self, source, *fields, attr: bool = False):
        selector = _selectors.select(*fields, attr=attr)
        super().__init__(source, selector)


@mark
class SelectAsTuple(Map):
    def __init__(self, source, *fields, attr: bool = False):
        selector = _selectors.select_as_tuple(*fields, attr=attr)
        super().__init__(source, selector)


@mark
class SelectAsDict(Map):
    def __init__(self, source, *fields, attr: bool = False, default=NoReturn):
        selector = _selectors.select_as_dict(*fields, attr=attr, default=default)
        super().__init__(source, selector)


@mark
class Reflect(Map):
    def __init__(self, source, mapping, attr: bool = False):
        selector = _selectors.reflect(mapping, attr=attr)
        super().__init__(source, selector)


@mark
class Flat(Query):
    def __init__(self, source, selector=None):
        super().__init__(source)
        self.selector = selector

    def _impl_iter(self):
        selector = self.selector
        if selector is None:
            return (_ for inner in self.source for _ in inner)
        else:
            return (_ for elm in self.source for _ in selector(elm))

    def _impl_aiter(self):
        selector = self.selector
        if selector is None:
            return (_ async for inner in self.source async for _ in inner)
        else:
            return (_ async for elm in self.source async for _ in selector(elm))


@mark
class FlatRecursive(Query):
    def __init__(self, source, selector):
        super().__init__(source)

        def scan(parent, selector):
            nodes = selector(parent)
            if nodes is None:
                return
            for node in nodes:
                yield node
                yield from scan(node, selector)

        self.scan = scan
        self.selector = selector

    def _impl_iter(self):
        scan = self.scan
        selector = self.selector
        for node in self.source:
            yield node
            yield from scan(node, selector)

    async def _impl_aiter(self):
        scan = self.scan
        selector = self.selector
        async for node in self.source:
            yield node
            for c in scan(node, selector):
                yield c


@mark
class PivotUnstack(Query):
    def __init__(self, source, default=None) -> None:
        super().__init__(source)
        self.default = default

    def _impl_iter(self):
        default = self.default
        dataframe = {}  # type: ignore
        data = []

        # 全てのカラムを取得
        for i, dic in enumerate(self.source):
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

        yield from dataframe.items()

    async def _impl_aiter(self):
        raise NotImplementedError()


@mark
class PivotStack(Query):
    def _impl_iter(self):
        data = dict(self.source.__iter__())
        columns = list(data.keys())

        for i in range(len(columns)):
            row = {}
            for c in columns:
                row[c] = data[c][i]

            yield row

    async def _impl_aiter(self):
        raise NotImplementedError()


@mark
class Enumerate(Query):
    def __init__(self, source, start: int = 0, step: int = 1):
        super().__init__(source)
        self.start = start
        self.step = step

    def _impl_iter(self):
        step = self.step
        return (x * step for x in enumerate(self.source, self.start))

    async def _impl_aiter(self):
        step = self.step
        i = self.start - step
        async for x in self.source:
            i += step
            yield i, x


@mark
class Request(Query):
    iter_type = IterType.BOTH

    def __init__(self, source, func, retry: int = 0) -> None:
        super().__init__(source)
        self.func = func
        self.retry = retry

        if retry:
            raise NotImplementedError("retry not implemented")

    def _impl_iter(self):
        func = self.func

        from .requests import Response, StopWatch

        for v in self.source:

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


@mark
class RequestAsync(Query):
    iter_type = IterType.ASYNC

    def __init__(self, source, func, timeout: float = None, retry: int = 0) -> None:
        super().__init__(source)
        self.func = func

        if retry:
            raise NotImplementedError("retry not implemented")

        if timeout:
            raise NotImplementedError("timeout not implemented")

    def _impl_iter(self):
        raise NotImplementedError()

    async def _impl_aiter(self):
        func = self.func

        from .requests import Response, StopWatch

        async for v in self.source:

            with StopWatch() as sw:
                err = None
                result = None
                try:
                    result = await func(**v)
                except Exception as e:
                    err = e

            res = Response(
                func, kwargs=v, err=err, result=result, start=sw.start, end=sw.end
            )

            yield res


@mark
class Parallel:
    def __init__(self, *args, **kwargs) -> None:
        pass


@mark
class Debug(Query):
    def __init__(self, source, breakpoint=lambda x: x, printer=print) -> None:
        super().__init__(source)
        self.breakpoint = breakpoint
        self.printer = printer

    def _impl_iter(self):
        breakpoint = self.breakpoint
        printer = self.printer

        for v in self.source:
            printer(v)
            breakpoint(v)
            yield v

    async def _impl_aiter(self):
        breakpoint = self.breakpoint
        printer = self.printer

        async for v in self.source:
            printer(v)
            breakpoint(v)
            yield v


@mark
class DebugPath(Query):
    """同期イテレータと非同期イテレータのどちらが実行されているか確認するデバッグ用のクエリです。"""

    def __init__(
        self, source, selector_sync=lambda x: -10, selector_async=lambda x: 10
    ) -> None:
        super().__init__(source)
        self.selector_sync = selector_sync
        self.selector_async = selector_async

    def _impl_iter(self):
        selector_sync = self.selector_sync
        for v in self.source:
            yield selector_sync(v)

    async def _impl_aiter(self):
        selector_async = self.selector_async
        async for v in self.source:
            yield selector_async(v)


@mark
class GroupBy(Query):
    def __init__(self, source, selector):
        super().__init__(source)
        self.selector = selector

    def _impl_iter(self):
        selector = self.selector
        results = defaultdict(list)
        for elm in self.source:
            k, v = selector(elm)
            results[k].append(v)

        for k, v in results.items():
            yield k, v

    async def _impl_aiter(self):
        selector = self.selector
        results = defaultdict(list)
        async for elm in self.source:
            k, v = selector(elm)
            results[k].append(v)

        for k, v in results.items():
            yield k, v


@mark
class Chunked(Query):
    def __init__(self, source, size: int) -> None:
        super().__init__(source)
        self.size = size

        if size <= 0:
            raise ValueError("count must be greater than 0")

    def _impl_iter(self):
        size = self.size
        current = 0
        running = True

        it = iter(self.source)

        while running:
            current = 0
            chunk = []

            while current < size:
                current += 1
                try:
                    val = next(it)
                    chunk.append(val)
                except StopIteration:
                    running = False
                    break

            if chunk:
                yield chunk

    async def _impl_aiter(self):
        size = self.size
        current = 0
        running = True

        it = self.source.__aiter__()

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


@mark
class Tee(Query):
    def __init__(self, source, size: int = 2) -> None:
        super().__init__(source)
        self.size = size

    def _impl_iter(self):
        return iter(itertools.tee(self.source, self.size))


@mark
class Filter(Query):
    def __init__(self, source, predicate):
        super().__init__(source)
        self.predicate = predicate

    def _impl_iter(self):
        return filter(self.predicate, self.source)

    async def _impl_aiter(self):
        predicate = self.predicate
        async for x in self.source:
            if predicate(x):
                yield x


@mark
class Must(Query):
    def __init__(self, source, predicate, msg: str = ""):
        super().__init__(source)
        self.predicate = predicate
        self.msg = msg

    def _impl_iter(self):
        predicate = self.predicate
        for elm in self.source:
            if not predicate(elm):
                raise MustError(f"{self.msg} {elm}")
            yield elm

    async def _impl_aiter(self):
        predicate = self.predicate
        async for elm in self.source:
            if not predicate(elm):
                raise MustError(f"{self.msg} {elm}")
            yield elm


@mark
class FilterType(Query):
    def __init__(self, source, *types):
        super().__init__(source)
        self.types = tuple(None.__class__ if x is None else x for x in types)

    def _impl_iter(self):
        types = self.types
        for elm in self.source:
            if isinstance(elm, types):
                yield elm

    async def _impl_aiter(self):
        types = self.types
        async for elm in self.source:
            if isinstance(elm, types):
                yield elm


@mark
class MustType(Query):
    def __init__(self, source, type, *types):
        super().__init__(source)
        self.types = (type, *types) if types else type

    def _impl_iter(self):
        types = self.types
        for elm in self.source:
            if not isinstance(elm, types):
                raise MustTypeError(f"{elm} is not {types}")
            yield elm

    async def _impl_aiter(self):
        types = self.types
        async for elm in self.source:
            if not isinstance(elm, types):
                raise MustTypeError(f"{elm} is not {types}")
            yield elm


@mark
class FilterUnique(Query):
    def __init__(self, source, selector=None):
        if selector is not None:
            source = MapNullable(source, selector)
        super().__init__(source)

    def _impl_iter(self):
        duplidate = set()
        for value in self.source:
            if value in duplidate:
                pass
            else:
                duplidate.add(value)
                yield value

    async def _impl_aiter(self):
        duplidate = set()
        async for value in self.source:
            if value in duplidate:
                pass
            else:
                duplidate.add(value)
                yield value


@mark
class MustUnique(Query):
    def __init__(self, source, selector=None):
        super().__init__(source)

        if selector is not None:
            self.selector = selector
        else:
            self.selector = lambda x: x

    def _impl_iter(self):
        selector = self.selector
        duplidate = set()
        for value in self.source:
            target = selector(value)
            if target in duplidate:
                raise DuplicateElementError(value)
            else:
                duplidate.add(target)
                yield value

    async def _impl_aiter(self):
        selector = self.selector
        duplidate = set()
        async for value in self.source:
            target = selector(value)
            if target in duplidate:
                raise DuplicateElementError(value)
            else:
                duplidate.add(target)
                yield value


def get_many_for_mapping(query_dict, keys):
    """"""
    undefined = object()
    for key in keys:
        obj = query_dict.get_or(key, undefined)
        if obj is not undefined:
            yield key, obj


def get_many_for_sequence(query_seq, keys):
    """"""
    undefined = object()
    for key in keys:
        obj = query_seq.get_or(key, undefined)
        if obj is not undefined:
            yield obj


def get_many_for_set(query_set, keys):
    """"""
    undefined = object()
    for key in keys:
        obj = query_set.get_or(key, undefined)
        if obj is not undefined:
            yield key


def get_duplicate_keys(keys):
    if len(keys) == len(set(keys)):
        return None

    from collections import Counter

    dup = []
    for k, count in Counter(keys).items():  # type: ignore
        if count > 1:
            dup.append(k)

    return dup


@mark
class FilterKeys(Query):
    def __init__(self, source, *keys):
        super().__init__(source)
        self.keys = keys

        dup = get_duplicate_keys(keys)
        if dup:
            raise ValueError("duplicate keys: {dup}")

        if isinstance(source, QueryDict):
            self.filter = get_many_for_mapping
        elif isinstance(source, QuerySeq):
            self.filter = get_many_for_sequence
        elif isinstance(source, QuerySet):
            self.filter = get_many_for_set
        else:
            raise TypeError(f"{source} is not QuerySeq, QueryDict or QuerySet")

    def _impl_iter(self):
        return self.filter(self.source, self.keys)

    async def _impl_aiter(self):
        for x in self._impl_iter():
            yield x


@mark
class MustKeys(Query):
    # Literal["set", "seq", "map"]
    def __init__(self, source, *keys, typ: str):
        super().__init__(source)
        self.keys = keys
        self.type = typ

        dup = get_duplicate_keys(keys)
        if dup:
            raise ValueError("duplicate keys: {dup}")

    def _impl_iter(self):
        source = self.source
        not_exists = set()
        key_values = []
        undefined = object()

        for k in self.keys:
            val = source.get_or(k, undefined)
            if val is undefined:
                not_exists.add(k)
            else:
                key_values.append((k, val))

        if not_exists:
            raise NotFoundError(str(not_exists))

        if self.type == "map":
            for k, v in key_values:
                yield k, v
        elif self.type == "seq":
            for k, v in key_values:
                yield v
        elif self.type == "set":
            for k, v in key_values:
                yield k
        else:
            raise TypeError(f"unknown type: {self.type}")


@mark
class Take(Query):
    def __init__(self, source, count_or_range: Union[int, range]):
        super().__init__(source)
        if isinstance(count_or_range, range):
            r = count_or_range
        else:
            r = range(count_or_range)
        self.start = r.start
        self.stop = r.stop
        self.step = r.step

        if self.start < 0:
            self.start = 0

        if self.stop < 0:
            self.stop = 0

        if self.step < 1:
            raise ValueError()

    def _impl_iter(self):
        return itertools.islice(self.source, self.start, self.stop, self.step)

    async def _impl_aiter(self):
        start = self.start
        stop = self.stop
        step = self.step

        current = 0

        it = self.source.__aiter__()

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


@mark
class TakeWhile(Query):
    def __init__(self, source, predicate):
        super().__init__(source)
        self.predicate = predicate

    def _impl_iter(self):
        predicate = self.predicate
        for v in self.source:
            if predicate(v):
                yield v
            else:
                break

    async def _impl_aiter(self):
        predicate = self.predicate
        async for v in self.source:
            if predicate(v):
                yield v
            else:
                break


@mark
class Skip(Query):
    def __init__(self, source, count_or_range: Union[int, range]):
        super().__init__(source)

        if isinstance(count_or_range, range):
            r = count_or_range
        else:
            r = range(count_or_range)
        self.start = r.start
        self.stop = r.stop

    def _impl_iter(self):
        start = self.start
        stop = self.stop

        current = 0

        it = iter(self.source)

        try:
            while current < start:
                yield next(it)
                current += 1
        except StopIteration:
            return

        try:
            while current < stop:
                next(it)
                current += 1
        except StopIteration:
            return

        for x in it:
            yield x

    async def _impl_aiter(self):
        start = self.start
        stop = self.stop

        current = 0

        it = self.source.__aiter__()

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


def take_page_calc(page: int, size: int):
    if size < 0:
        raise ValueError("size must be >= 0")
    start = (page - 1) * size
    stop = start + size
    return start, stop


@mark
class TakePage(Take):
    def __init__(self, source, page: int, size: int):
        start, stop = take_page_calc(page, size)
        super().__init__(source, range(start, stop))


@mark
class OrderByMap(Query):
    def __init__(self, source, selector=None, desc: bool = False) -> None:
        super().__init__(source)
        self.selector = selector
        self.desc = desc

    def _impl_iter(self):
        yield from sorted(self.source, key=self.selector, reverse=self.desc)

    async def _impl_aiter(self):
        raise NotImplementedError()


@mark
class OrderBy(OrderByMap):
    def __init__(self, source, *fields, desc: bool = False, attr: bool = False) -> None:
        if not len(fields):
            selector = None
        else:
            if attr:
                selector = attrgetter(*fields)
            else:
                selector = itemgetter(*fields)

        super().__init__(source, selector=selector, desc=desc)


@mark
class OrderByReverse(Query):
    def _impl_iter(self):
        source = self.source
        if hasattr(source, "__reversed__"):
            if isinstance(source, Mapping):
                return ((k, source[k]) for k in reversed(source))  # type: ignore
            else:
                return reversed(source)
        else:
            return reversed(list(source))

    def _impl_aiter(self):
        raise NotImplementedError()


@mark
class OrderByShuffle(OrderByMap):
    def _impl_iter(self):
        import random

        yield from sorted(self.source, key=lambda k: random.random())

    def _impl_aiter(self):
        raise NotImplementedError()


@mark
class Cartesian(Query):
    def __init__(self, source, *iterables):
        super().__init__(source)
        self.iterables = iterables

    def _impl_iter(self):
        import itertools

        return itertools.product(self.source, *self.iterables)

    def _impl_aiter(self):
        raise NotImplementedError()
