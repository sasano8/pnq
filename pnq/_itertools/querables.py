from typing import Callable, NoReturn, Union

from pnq._itertools.common import name_as

from .. import selectors
from . import _async as A
from . import _sync as S


def no_implement(*args, **kwargs):
    raise NotImplementedError()


class Staticmethod:
    def __or__(self, other):
        return staticmethod(other)


sm = Staticmethod()


#################################
# query
#################################
class Query:
    _iter_type = 0
    # _sit: Callable
    # _ait: Callable

    def __init__(self, source):
        self.source = source

    def __iter__(self, *args, **kwds):
        return self._sit(*args, **kwds)

    def __aiter__(self, *args, **kwds):
        typ, it = self.__iter_or_aiter__()
        if typ == 0:
            self._sit(*args, **kwds)
        elif typ == 1:
            self._ait(*args, **kwds)
        else:
            raise TypeError("{!r} is not iterable".format(self.source))

    def __iter_or_aiter__(self):
        get_iter = getattr(self.source, "__iter_or_aiter__", None)
        if get_iter:
            return get_iter()
        else:
            if hasattr(self.source, "__aiter__"):
                return 1, self.source.__aiter__()
            else:
                return 0, self.source.__iter__()

    def __args__(self):
        return tuple(), {}

    def _impl_iter(self):
        args, kwargs = self.__args__()
        return self._sit(self.source, *args, **kwargs)

    def _impl_aiter(self):
        args, kwargs = self.__args__()
        return self._ait(self.source, *args, **kwargs)


class Map(Query):
    _ait = sm | A.queries._map
    _sit = sm | S.queries._map

    def __init__(self, source, selector, unpack=""):
        super().__init__(source)
        self.selector = selector
        self.unpack = unpack

    def __args__(self):
        return (self.selector, self.unpack), {}


class Select(Map):
    def __init__(self, source, *args, attr=False):
        selector = selectors.select(*args, attr=attr)
        super().__init__(source, selector)


class SelectAsTuple(Map):
    def __init__(self, source, *args, attr=False):
        selector = selectors.select_as_tuple(*args, attr=attr)
        super().__init__(source, selector)


class SelectAsDict(Map):
    def __init__(self, source, *args, attr=False, default=NoReturn):
        selector = selectors.select_as_dict(*args, attr=attr, default=default)
        super().__init__(source, selector)


class Reflect(Map):
    def __init__(self, source, mapping, attr=False):
        selector = selectors.reflect(mapping, attr=attr)
        super().__init__(source, selector)


class MapAwait(Query):
    _ait = sm | A.queries.map_await
    _sit = sm | S.queries.map_await


class Flat(Query):
    _ait = sm | A.queries.flat
    _sit = sm | S.queries.flat

    def __init__(self, source, selector=None):
        super().__init__(source)
        self.selector = selector

    def __args__(self):
        return (self.selector,), {}


class FlatRecursive(Query):
    _ait = sm | A.queries.flat_recursive
    _sit = sm | S.queries.flat_recursive

    def __init__(self, source, selector):
        super().__init__(source)
        self.selector = selector

    def __args__(self):
        return (self.selector,), {}


class PivotUnstack(Query):
    _ait = sm | A.queries.pivot_unstack
    _sit = sm | S.queries.pivot_unstack

    def __init__(self, source, default=None):
        super().__init__(source)
        self.default = default

    def __args__(self):
        return (self.default,), {}


class Pivotstack(Query):
    _ait = sm | A.queries.pivot_stack
    _sit = sm | S.queries.pivot_stack


class Enumerate(Query):
    _ait = sm | A.queries._enumerate
    _sit = sm | S.queries._enumerate

    def __init__(self, source, start=0, step=1):
        super().__init__(source)
        self.start = start
        self.step = step

    def __args__(self):
        return (self.start, self.step), {}


class GroupBy(Query):
    _ait = sm | A.queries.group_by
    _sit = sm | S.queries.group_by

    def __init__(self, source, selector=None):
        super().__init__(source)
        self.selector = selector

    def __args__(self):
        return (self.selector), {}


class Join(Query):
    _ait = sm | A.queries.join
    _sit = sm | S.queries.join


class GroupJoin(Query):
    _ait = sm | A.queries.group_join
    _sit = sm | S.queries.group_join


class Chunked(Query):
    _ait = sm | A.queries.chunked
    _sit = sm | S.queries.chunked

    def __init__(self, source, size: int):
        super().__init__(source)
        self.size = size

        if size <= 0:
            raise ValueError("count must be greater than 0")

    def __args__(self):
        return (self.size), {}


class Tee(Query):
    _ait = sm | A.queries.tee
    _sit = sm | S.queries.tee

    def __init__(self, source, size: int):
        super().__init__(source)
        self.size = size

        if size <= 0:
            raise ValueError("count must be greater than 0")

    def __args__(self):
        return (self.size), {}


class Request(Query):
    _ait = sm | A.queries.request
    _sit = sm | S.queries.request

    def __init__(self, source, func, timeout: float = None, retry: int = 0):
        super().__init__(source)
        self.func = func
        self.retry = retry
        self.timeout = timeout

    def __args__(self):
        return (self.func, self.timeout, self.retry), {}


class RequestAsync(Query):
    _ait = sm | A.queries.request_async
    _sit = sm | S.queries.request_async

    def __init__(self, source, func, timeout: float = None, retry: int = 0):
        super().__init__(source)
        self.func = func
        self.retry = retry

    def __args__(self):
        return (self.func, self.timeout, self.retry), {}


# PEP 3148
class Parallel(Query):
    _ait = sm | A.queries.parallel
    _sit = sm | S.queries.parallel


class Debug(Query):
    _ait = sm | A.queries.debug
    _sit = sm | S.queries.debug


class DebugPath(Query):
    def __init__(self, source, func=lambda x: -10, async_func=lambda x: 10):
        super().__init__(source)
        self.func = func
        self.async_func = async_func

    def __args__(self):
        return (self.func, self.async_func), {}

    def _impl_iter(self):
        func = self.func
        for v in self.source:
            yield func(v)

    async def _impl_aiter(self):
        async_func = self.async_func
        async for v in self.source:
            yield async_func(v)


class UnionAll(Query):
    _ait = sm | A.queries.union_all
    _sit = sm | S.queries.union_all


# class Extend(Query):
#     _ait = A.queries.extend
#     _sit = S.queries.extend


@name_as("Union")
class _Union(Query):
    _ait = sm | A.queries.union
    _sit = sm | S.queries.union


class UnionIntersect(Query):
    _ait = sm | A.queries.union_intersect
    _sit = sm | S.queries.union_intersect


class UnionMinus(Query):
    _ait = sm | A.queries.union_intersect
    _sit = sm | S.queries.union_intersect


class Zip(Query):
    _ait = sm | no_implement
    _sit = sm | S.queries._zip


class Compress(Query):
    _ait = sm | A.queries.compress
    _sit = sm | S.queries.compress


class Cartesian(Query):
    _ait = sm | A.queries.cartesian
    _sit = sm | S.queries.cartesian


class Filter(Query):
    _ait = sm | A.queries._filter
    _sit = sm | S.queries._filter

    def __init__(self, source, predicate, unpack=""):
        super().__init__(source)
        self.predicate = predicate
        self.unpack = unpack

    def __args__(self):
        return (self.predicate, self.unpack), {}


class Must(Query):
    _ait = sm | A.queries.must
    _sit = sm | S.queries.must

    def __init__(self, source, predicate, msg: str = ""):
        super().__init__(source)
        self.predicate = predicate
        self.msg = msg

    def __args__(self):
        return (self.predicate, self.msg), {}


class FilterType(Query):
    _ait = sm | A.queries.filter_type
    _sit = sm | S.queries.filter_type

    def __init__(self, source, *types):
        super().__init__(source)
        self.types = tuple(None.__class__ if x is None else x for x in types)

    def __args__(self):
        return self.types, {}


class MustType(Query):
    _ait = sm | A.queries.must_type
    _sit = sm | S.queries.must_type

    def __init__(self, source, *types):
        super().__init__(source)
        self.types = tuple(None.__class__ if x is None else x for x in types)

    def __args__(self):
        return self.types, {}


class FilterUnique(Query):
    _ait = sm | A.queries.filter_unique
    _sit = sm | S.queries.filter_unique

    def __init__(self, source, selector=None):
        super().__init__(source)
        self.selector = selector

    def __args__(self):
        return self.selector, {}


class MustUnique(Query):
    _ait = sm | A.queries.must_unique
    _sit = sm | S.queries.must_unique

    def __init__(self, source, selector=None):
        super().__init__(source)
        self.selector = selector

    def __args__(self):
        return self.selector, {}


class FilterKeys(Query):
    _ait = sm | A.queries.filter_keys
    _sit = sm | S.queries.filter_keys


class MustKeys(Query):
    _ait = sm | A.queries.must_keys
    _sit = sm | S.queries.must_keys


class Take(Query):
    _ait = sm | A.queries.take
    _sit = sm | S.queries.take

    def __init__(self, source, count_or_range: Union[int, range]):
        super().__init__(source)
        if isinstance(count_or_range, range):
            r = count_or_range
        else:
            r = range(count_or_range)

        if r.start < 0:
            raise ValueError()

        if r.stop < 0:
            raise ValueError()

        if r.step < 1:
            raise ValueError()

        self.r = r

    def __args__(self):
        return (self.r,), {}


class Skip(Take):
    _ait = sm | A.queries.skip
    _sit = sm | S.queries.skip


class TakePage(Take):
    _ait = sm | A.queries.take
    _sit = sm | S.queries.take

    def __init__(self, source, page: int, size: int):
        if not page >= 1:
            raise ValueError("page must be >= 1")
        if size < 0:
            raise ValueError("size must be >= 0")
        start = (page - 1) * size
        stop = start + size
        super().__init__(source, range(start, stop))


class TakeWhile(Query):
    _ait = sm | A.queries.take_while
    _sit = sm | S.queries.take_while

    def __init__(self, source, predicate):
        super().__init__(source)
        self.predicate = predicate

    def __args__(self):
        return (self.predicate,), {}


class SkipWhile(TakeWhile):
    _ait = sm | A.queries.skip_while
    _sit = sm | S.queries.skip_while


class OrderBy(Query):
    _ait = sm | A.queries.order_by
    _sit = sm | S.queries.order_by

    def __init__(self, source, selector=None, desc: bool = False):
        super().__init__(source)
        self.selector = selector
        self.desc = desc

    def __args__(self):
        return (self.selector, self.desc), {}


class OrderBySelect(OrderBy):
    _ait = sm | A.queries.order_by
    _sit = sm | S.queries.order_by

    def __init__(self, source, *fields, attr: bool = False, desc: bool = False):
        if not len(fields):
            selector = None
        else:
            if attr:
                selector = selectors.select_from_attr(*fields)
            else:
                selector = selectors.select_from_item(*fields)

        super().__init__(source, selector, desc)


class OrderByReverse(Query):
    _ait = sm | A.queries.order_by_reverse
    _sit = sm | S.queries.order_by_reverse


class OrderByShuffle(Query):
    _ait = sm | A.queries.order_by_shuffle
    _sit = sm | S.queries.order_by_shuffle


#################################
# finalizer
#################################
class FinalizerBase:
    exists = S.finalizers.exists
    len = S.finalizers._len
    all = S.finalizers._all
    any = S.finalizers._any
    sum = S.finalizers._sum
    min = S.finalizers._min
    max = S.finalizers._max
    average = S.finalizers.average
    reduce = S.finalizers.reduce
    contains = S.finalizers.contains
    concat = S.finalizers.concat
    each = S.finalizers.each
    one = S.finalizers.one
    one_or = S.finalizers.one_or
    one_or_raise = S.finalizers.one_or_raise
    first = S.finalizers.first
    first_or = S.finalizers.first_or
    first_or_raise = S.finalizers.first_or_raise
    last = S.finalizers.last
    last_or = S.finalizers.last_or
    last_or_raise = S.finalizers.last_or_raise


class AsyncFinalizerBase:
    exists = A.finalizers.exists
    len = A.finalizers._len
    all = A.finalizers._all
    any = A.finalizers._any
    sum = A.finalizers._sum
    min = A.finalizers._min
    max = A.finalizers._max
    average = A.finalizers.average
    reduce = A.finalizers.reduce
    contains = A.finalizers.contains
    concat = A.finalizers.concat
    one = A.finalizers.one
    one_or = A.finalizers.one_or
    one_or_raise = A.finalizers.one_or_raise
    first = A.finalizers.first
    first_or = A.finalizers.first_or
    first_or_raise = A.finalizers.first_or_raise
    last = A.finalizers.last
    last_or = A.finalizers.last_or
    last_or_raise = A.finalizers.last_or_raise
    each = A.finalizers.each


class Finalizer(FinalizerBase):
    def __init__(self, source):
        self.source = source

    def __iter__(self):
        return self.source.__iter__()

    def result(self, timeout: float = None):
        return list(self.source)


class AsyncFinalizer(AsyncFinalizerBase):
    def __init__(self, source):
        self.source = source

    def __aiter__(self):
        return self.source.__aiter__()

    def __await__(self):
        return self._result_async().__await__()

    async def _result_async(self):
        return [x async for x in self]
