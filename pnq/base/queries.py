import asyncio
from collections import defaultdict
from operator import attrgetter, itemgetter
from typing import Callable, NoReturn

from .core import IterType, Query

exports = []


def mark(func):
    exports.append(func)
    return func


@mark
class Lazy(Query):
    def __init__(self, source, finalizer: Callable, *args, **kwargs):
        super().__init__(source)
        self.finalizer = finalizer
        self.args = args
        self.kwargs = kwargs

    def __call__(self):
        return self.finalizer(self.source, *self.args, **self.kwargs)

    def __await__(self):
        return self.finalizer(self.source, *self.args, **self.kwargs).__await__()


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
class Map(Query):
    def __init__(self, source, selector):
        super().__init__(source)
        self.selector = selector

    def _impl_iter(self):
        selector = self.selector
        for x in self.source:
            yield selector(x)

    async def _impl_aiter(self):
        selector = self.selector
        async for x in self.source:
            yield selector(x)


@mark
class UnpackPos(Map):
    def _impl_iter(self):
        selector = self.selector
        for v in self.source:
            yield selector(*v)

    async def _impl_aiter(self):
        selelctor = self.selector
        async for v in self.source:
            yield selelctor(*v)


@mark
class UnpackKw(Map):
    def _impl_iter(self):
        selector = self.selector
        for v in self.source:
            yield selector(**v)

    async def _impl_aiter(self):
        selector = self.selector
        async for v in self.source:
            yield selector(**v)


@mark
class Select(Map):
    def __init__(self, source, *fields, attr: bool = False):
        if attr:
            selector = attrgetter(*fields)
        else:
            selector = itemgetter(*fields)
        super().__init__(source, selector)


@mark
class SelectAsTuple(Map):
    def __init__(self, source, *fields, attr: bool = False):
        if len(fields) > 1:
            if attr:
                selector = attrgetter(*fields)
            else:
                selector = itemgetter(*fields)
        elif len(fields) == 1:
            field = fields[0]
            if attr:
                selector = lambda x: (getattr(x, field),)
            else:
                selector = lambda x: (x[field],)
        else:
            selector = lambda x: ()
        super().__init__(source, selector)


@mark
class SelectAsDict(Map):
    def __init__(self, source, *fields, attr: bool = False, default=NoReturn):
        if attr:
            if default is NoReturn:
                selector = lambda x: {k: getattr(x, k) for k in fields}  # noqa
            else:
                selector = lambda x: {k: getattr(x, k, default) for k in fields}  # noqa
        else:
            if default is NoReturn:
                selector = lambda x: {k: x[k] for k in fields}  # noqa
            else:

                def get(obj, k):
                    try:
                        return obj[k]
                    except Exception:
                        return default

                selector = lambda x: {k: get(x, k) for k in fields}  # noqa
        super().__init__(source, selector)


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
class Pivotstack(Query):
    def _impl_iter(self):
        data = dict(self.source)
        columns = list(data.keys())

        for i in range(len(columns)):
            row = {}
            for c in columns:
                row[c] = data[c][i]

            yield row

    async def _impl_aiter(self):
        raise NotImplementedError()


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

        from ..requests import Response, StopWatch

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

        from ..requests import Response, StopWatch

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
class DebugMap(Query):
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
    def __init__(self, source, predicate):
        super().__init__(source)
        self.predicate = predicate

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
        self.types = types

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
    def __init__(self, source, *types):
        super().__init__(source)
        self.types = types

    def _impl_iter(self):
        types = self.types
        for elm in self.source:
            if not isinstance(elm, types):
                raise MustTypeError(f"{elm} is not {tuple(x.__name__ for x in types)}")
            yield elm

    async def _impl_aiter(self):
        types = self.types
        async for elm in self.source:
            if not isinstance(elm, types):
                raise MustTypeError(f"{elm} is not {tuple(x.__name__ for x in types)}")
            yield elm


@mark
class FilterUnique(Query):
    def __init__(self, source, selector=None):
        if selector is not None:
            source = Map(source, selector)
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


@mark
class Take(Query):
    def __init__(self, source, count: int):
        super().__init__(source)
        self.count = count

    def _impl_iter(self):
        count = self.count
        current = 0

        it = iter(self.source)

        try:
            while current < count:
                yield next(it)
                current += 1
        except StopIteration:
            return

    async def _impl_aiter(self):
        raise NotImplementedError()


@mark
class TakeRange(Query):
    def __init__(self, source, start: int = 0, stop: int = None):
        super().__init__(source)

        if start < 0:
            start = 0

        if stop is None:
            stop = float("inf")
        elif stop < 0:
            stop = 0
        else:
            pass

        self.start = start
        self.stop = stop

    def _impl_iter(self):
        start = self.start
        stop = self.stop

        current = 0

        it = iter(self.source)

        try:
            while current < start:
                next(it)
                current += 1
        except StopIteration:
            yield from ()

        try:
            while current < stop:
                yield next(it)
                current += 1
        except StopIteration:
            pass

    async def _impl_aiter(self):
        raise NotImplementedError()


def take_page_calc(page: int, size: int):
    if size < 0:
        raise ValueError("size must be >= 0")
    start = (page - 1) * size
    stop = start + size
    return start, stop


@mark
class TakePage(TakeRange):
    def __init__(self, source, page: int, size: int):
        start, stop = take_page_calc(page, size)
        super().__init__(source, start=start, stop=stop)


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
class OrderByShuffle(OrderByMap):
    def _impl_iter(self):
        import random

        yield from sorted(self.source, key=lambda k: random.random())
