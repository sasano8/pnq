from typing import (
    TYPE_CHECKING,
    AbstractSet,
    AsyncIterable,
    Callable,
    Generic,
    Iterable,
    List,
    Mapping,
    NoReturn,
    Sequence,
    TypeVar,
    Union,
)

from pnq._itertools.common import name_as
from pnq.exceptions import NotFoundError
from pnq.types import Arguments

from .. import selectors
from . import _async as A
from . import _sync as S
from .core import IterType
from .core import Query as QueryBase
from .core import QueryDict, QuerySeq, QuerySet


def no_implement(*args, **kwargs):
    raise NotImplementedError()


F = TypeVar("F", bound=Callable)


class Staticmethod:
    def __or__(self, other: F) -> F:
        return staticmethod(other)  # type: ignore


exports = []


def export(cls):
    exports.append(cls)
    return cls


T = TypeVar("T")
sm = Staticmethod()


#################################
# query
#################################
class Query(QueryBase[T]):
    _iter_type = IterType.BOTH

    if TYPE_CHECKING:
        _sit: Callable
        _ait: Callable

    def __init__(self, source: Union[Iterable[T], AsyncIterable[T]]):
        super().__init__(source)

    _args = Arguments()

    def _impl_iter(self):
        x = self._args
        return self._sit(self.source, *x.args, **x.kwargs)

    def _impl_aiter(self):
        x = self._args
        return self._ait(self.source, *x.args, **x.kwargs)


@export
class Map(Query):
    _ait = sm | A.queries._map
    _sit = sm | S.queries._map

    def __init__(self, source, selector, unpack=""):
        super().__init__(source)
        self._args = Arguments(selector, unpack)


@export
class MapNullable(Map):
    ...


@export
class UnpackPos(Map):
    def __init__(self, source, selector):
        super().__init__(source, selector, unpack="*")


@export
class UnpackKw(Map):
    def __init__(self, source, selector):
        super().__init__(source, selector, unpack="**")


@export
class Select(Map):
    def __init__(self, source, *args, attr=False):
        selector = selectors.select(*args, attr=attr)
        super().__init__(source, selector)


@export
class SelectAsTuple(Map):
    def __init__(self, source, *args, attr=False):
        selector = selectors.select_as_tuple(*args, attr=attr)
        super().__init__(source, selector)


@export
class SelectAsDict(Map):
    def __init__(self, source, *args, attr=False, default=NoReturn):
        selector = selectors.select_as_dict(*args, attr=attr, default=default)
        super().__init__(source, selector)


@export
class Reflect(Map):
    def __init__(self, source, mapping, attr=False):
        selector = selectors.reflect(mapping, attr=attr)
        super().__init__(source, selector)


@export
class Gather(Query):
    _ait = sm | A.queries.gather
    _sit = sm | S.queries.gather


@export
class Flat(Query):
    _ait = sm | A.queries.flat
    _sit = sm | S.queries.flat

    def __init__(self, source, selector=None):
        super().__init__(source)
        self.selector = selector
        self._args = Arguments(selector)


@export
class FlatRecursive(Query):
    _ait = sm | A.queries.flat_recursive
    _sit = sm | S.queries.flat_recursive

    def __init__(self, source, selector):
        super().__init__(source)
        self._args = Arguments(selector)


@export
class PivotUnstack(Query):
    _ait = sm | A.queries.pivot_unstack
    _sit = sm | S.queries.pivot_unstack

    def __init__(self, source, default=None):
        super().__init__(source)
        self._args = Arguments(default)


@export
class PivotStack(Query):
    _ait = sm | A.queries.pivot_stack
    _sit = sm | S.queries.pivot_stack


@export
class Enumerate(Query):
    _ait = sm | A.queries._enumerate
    _sit = sm | S.queries._enumerate

    def __init__(self, source, start=0, step=1):
        super().__init__(source)
        self._args = Arguments(start, step)


@export
class GroupBy(Query):
    _ait = sm | A.queries.group_by
    _sit = sm | S.queries.group_by

    def __init__(self, source, selector=None):
        super().__init__(source)
        self._args = Arguments(selector)


@export
class Join(Query):
    _ait = sm | A.queries.join
    _sit = sm | S.queries.join


@export
class GroupJoin(Query):
    _ait = sm | A.queries.group_join
    _sit = sm | S.queries.group_join


@export
class Chunked(Query):
    _ait = sm | A.queries.chunked
    _sit = sm | S.queries.chunked

    def __init__(self, source, size: int):
        super().__init__(source)
        if size <= 0:
            raise ValueError("count must be greater than 0")

        self._args = Arguments(size)


@export
class Tee(Query):
    _ait = sm | A.queries.tee
    _sit = sm | S.queries.tee

    def __init__(self, source, size: int):
        super().__init__(source)
        if size <= 0:
            raise ValueError("count must be greater than 0")

        self._args = Arguments(size)


@export
class Request(Query):
    # _ait = sm | A.queries.request
    # _sit = sm | S.queries.request
    _ait = sm | A.concurrent.request
    _sit = sm | S.concurrent.request

    # def __init__(self, source, func, timeout: float = None, retry: int = 0):
    #     super().__init__(source)
    #     self._args = Arguments(func, retry)

    def __init__(
        self,
        source: AsyncIterable[T],
        func,
        executor=None,
        *,
        unpack="",
        chunksize=1,
        retry: int = None,
        timeout: float = None,
    ):
        super().__init__(source)
        self._args = Arguments(
            func,
            executor,
            unpack=unpack,
            chunksize=chunksize,
            retry=retry,
            timeout=timeout,
        )


@export
class Parallel(Query):
    """
    PEP 3148
    I/Oバウンドを効率化するにはchunksizeを1にする。
    CPUをフル活用するにはchunksizeを大きくする。
    ProcessPoolのみchunksizeは有効
    スクレイピングなどの重い処理を並列化する parallel([], func, chunksize=1)
    簡単な計算を大量に行う処理を並列化する parallel([], func, chunksize=100)
    """

    _ait = sm | A.concurrent.parallel
    _sit = sm | S.concurrent.parallel

    def __init__(self, source, func, executor=None, *, unpack="", chunksize=1):
        super().__init__(source)
        self._args = Arguments(func, executor, unpack=unpack, chunksize=chunksize)


@export
class Debug(Query):
    _ait = sm | A.queries.debug
    _sit = sm | S.queries.debug

    def __init__(self, source, breakpoint=lambda x: x, printer=print):
        super().__init__(source)
        self._args = Arguments(breakpoint, printer)


@export
class DebugPath(Query):
    def __init__(self, source, func=lambda x: -10, async_func=lambda x: 10):
        super().__init__(source)
        self._args = Arguments(func, async_func)

    def _impl_iter(self):
        func = self._args.args[0]
        for v in self.source:
            yield func(v)

    async def _impl_aiter(self):
        async_func = self._args.args[1]
        async for v in self.source:
            yield async_func(v)


@export
class UnionAll(Query):
    _ait = sm | A.queries.union_all
    _sit = sm | S.queries.union_all


# class Extend(Query):
#     _ait = A.queries.extend
#     _sit = S.queries.extend


@export
@name_as("Union")
class _Union(Query):
    _ait = sm | A.queries.union
    _sit = sm | S.queries.union


@export
class UnionIntersect(Query):
    _ait = sm | A.queries.union_intersect
    _sit = sm | S.queries.union_intersect


@export
class UnionMinus(Query):
    _ait = sm | A.queries.union_intersect
    _sit = sm | S.queries.union_intersect


@export
class Zip(Query):
    _ait = sm | no_implement
    _sit = sm | S.queries._zip


@export
class Compress(Query):
    _ait = sm | A.queries.compress
    _sit = sm | S.queries.compress


@export
class Cartesian(Query):
    _ait = sm | A.queries.cartesian
    _sit = sm | S.queries.cartesian

    def __init__(self, source, *iterables):
        super().__init__(source)
        self._args = Arguments.from_obj(iterables, {})


@export
class Filter(Query):
    _ait = sm | A.queries._filter
    _sit = sm | S.queries._filter

    def __init__(self, source, predicate, unpack=""):
        super().__init__(source)
        self._args = Arguments(predicate, unpack)


@export
class Must(Query):
    _ait = sm | A.queries.must
    _sit = sm | S.queries.must

    def __init__(self, source, predicate, msg: str = ""):
        super().__init__(source)
        self._args = Arguments(predicate, msg)


@export
class FilterType(Query):
    _ait = sm | A.queries.filter_type
    _sit = sm | S.queries.filter_type

    def __init__(self, source, *types):
        super().__init__(source)
        types = tuple(None.__class__ if x is None else x for x in types)
        self._args = Arguments.from_obj(types, {})


@export
class MustType(Query):
    _ait = sm | A.queries.must_type
    _sit = sm | S.queries.must_type

    def __init__(self, source, *types):
        super().__init__(source)
        types = tuple(None.__class__ if x is None else x for x in types)
        self._args = Arguments.from_obj(types, {})


@export
class FilterUnique(Query):
    _ait = sm | A.queries.filter_unique
    _sit = sm | S.queries.filter_unique

    def __init__(self, source, selector=None):
        super().__init__(source)
        self._args = Arguments(selector)


@export
class MustUnique(Query):
    _ait = sm | A.queries.must_unique
    _sit = sm | S.queries.must_unique

    def __init__(self, source, selector=None):
        super().__init__(source)
        self._args = Arguments(selector)


def get_many_for_mapping(query_dict, keys):
    """"""
    undefined = object()
    for key in keys:
        obj = query_dict.get(key, undefined)
        if obj is not undefined:
            yield key, obj


def get_many_for_sequence(query_seq, keys):
    """"""
    for key in keys:
        try:
            yield query_seq[key]
        except IndexError:
            ...


def get_many_for_set(query_set, keys):
    """"""
    for key in keys:
        if key in query_set:
            yield key


@export
class FilterKeys(Query):
    _ait = sm | A.queries.filter_keys
    _sit = sm | S.queries.filter_keys

    def __init__(self, source, *keys):
        super().__init__(source)
        keys = dict.fromkeys(keys, None)  # use dict. because set has no order.
        self.keys = keys

        if isinstance(self.source, (QuerySeq, QueryDict, QuerySet)):
            source = self.source.source
        else:
            source = self.source
        if isinstance(source, Mapping):
            filter = get_many_for_mapping
        elif isinstance(source, Sequence):
            filter = get_many_for_sequence
        elif isinstance(source, AbstractSet):
            filter = get_many_for_set
        else:
            raise TypeError(f"{source} is not QuerySeq, QueryDict or QuerySet")

        self._ref = source
        self._filter = filter
        self._args = Arguments.from_obj(keys, {})

    def _impl_iter(self):
        return self._filter(self._ref, self.keys)

    async def _impl_aiter(self):
        for x in self._impl_iter():
            yield x


def get_for_dict(obj: dict, k, default):
    return obj.get(k, default)


def get_for_seq(obj: list, k, default):
    try:
        return obj[k]
    except IndexError:
        return default


def get_for_set(obj: set, k, default):
    if k in obj:
        return k
    else:
        return default


@export
class MustKeys(Query):
    _ait = sm | A.queries.must_keys
    _sit = sm | S.queries.must_keys

    def __init__(self, source, *keys, typ: str):
        # TODO: remove typ

        super().__init__(source)
        keys = dict.fromkeys(keys, None)  # use dict. because set has no order.
        self.keys = keys
        self._args = Arguments.from_obj(keys, {})

        if isinstance(self.source, (QuerySeq, QueryDict, QuerySet)):
            source = self.source.source
        else:
            source = self.source
        if isinstance(source, Mapping):
            getter = get_for_dict
        elif isinstance(source, Sequence):
            getter = get_for_seq
        elif isinstance(source, AbstractSet):
            getter = get_for_set
        else:
            raise TypeError(f"{source} is not QuerySeq, QueryDict or QuerySet")

        self._ref = source
        self._getter = getter

    def _impl_iter(self):
        source = self._ref
        not_exists = set()
        key_values = []
        undefined = object()
        getter = self._getter

        for k in self.keys:
            val = getter(source, k, undefined)
            if val is undefined:
                not_exists.add(k)
            else:
                key_values.append((k, val))

        if not_exists:
            raise NotFoundError(str(not_exists))

        if getter is get_for_dict:
            for k, v in key_values:
                yield k, v
        elif getter == get_for_seq:
            for k, v in key_values:
                yield v
        elif getter == get_for_set:
            for k, v in key_values:
                yield k
        else:
            raise TypeError(f"unknown type")

    async def _impl_aiter(self):
        for x in self._impl_iter():
            yield x


@export
class Take(Query):
    _ait = sm | A.queries.take
    _sit = sm | S.queries.take

    def __init__(self, source, count_or_range: Union[int, range]):
        super().__init__(source)
        if isinstance(count_or_range, range):
            r = range(
                max(count_or_range.start, 0),
                max(count_or_range.stop, 0),
                max(count_or_range.step, 1),
            )
        else:
            if count_or_range < 0:
                count_or_range = 0

            r = range(count_or_range)

        self._args = Arguments(r)


@export
class Skip(Take):
    _ait = sm | A.queries.skip
    _sit = sm | S.queries.skip


@export
class TakePage(Take):
    _ait = sm | A.queries.take
    _sit = sm | S.queries.take

    def __init__(self, source, page: int, size: int):
        if page < 1:
            raise ValueError("page must be >= 1")
        if size < 0:
            raise ValueError("size must be >= 0")
        start = (page - 1) * size
        stop = start + size
        super().__init__(source, range(start, stop))


@export
class TakeWhile(Query):
    _ait = sm | A.queries.take_while
    _sit = sm | S.queries.take_while

    def __init__(self, source, predicate):
        super().__init__(source)
        self._args = Arguments(predicate)


@export
class SkipWhile(TakeWhile):
    _ait = sm | A.queries.skip_while
    _sit = sm | S.queries.skip_while


@export
class OrderByMap(Query):
    _ait = sm | A.queries.order_by
    _sit = sm | S.queries.order_by

    def __init__(self, source, selector=None, desc: bool = False):
        super().__init__(source)
        self._args = Arguments(selector, desc)


@export
class OrderBy(OrderByMap):
    _ait = sm | A.queries.order_by
    _sit = sm | S.queries.order_by

    def __init__(self, source, *fields, desc: bool = False, attr: bool = False):
        if not len(fields):
            selector = None
        else:
            if attr:
                selector = selectors.select_from_attr(*fields)
            else:
                selector = selectors.select_from_item(*fields)
        super().__init__(source, selector, desc)


@export
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


@export
class OrderByReverse(Query):
    _ait = sm | A.queries.order_by_reverse
    _sit = sm | S.queries.order_by_reverse


@export
class OrderByShuffle(Query):
    _ait = sm | A.queries.order_by_shuffle
    _sit = sm | S.queries.order_by_shuffle


@export
class Sleep(Query):
    _ait = sm | A.queries.sleep
    _sit = sm | S.queries.sleep

    def __init__(self, source, seconds):
        super().__init__(source)
        self._args = Arguments(seconds)


# いらない！！！
@export
class AsyncMap(Query):
    iter_type = IterType.ASYNC

    def __init__(self, source, selector):
        super().__init__(source)
        self.selector = selector

    async def _impl_aiter(self):
        selector = self.selector
        async for x in self.source:
            yield await selector(x)


# TODO： いらない！！！
@export
class Lazy(Query):
    def __init__(self, source, finalizer, *args, **kwargs):
        super().__init__(source)
        self.finalizer = finalizer
        self.args = args
        self.kwargs = kwargs

    def __call__(self):
        return self.finalizer(self.source, *self.args, **self.kwargs)

    def __await__(self):
        coro = self.finalizer(self.source, *self.args, **self.kwargs)
        return coro.__await__()


#################################
# finalizer
#################################
class FinalizerBase:
    empty = S.finalizers.empty
    exists = S.finalizers.exists
    len = S.finalizers._len
    all = S.finalizers._all
    any = S.finalizers._any
    sum = S.finalizers._sum
    min = S.finalizers._min
    max = S.finalizers._max
    average = S.finalizers.average
    reduce = S.finalizers.reduce
    accumulate = S.finalizers.accumulate
    find = S.finalizers.find
    contains = S.finalizers.contains  # TODO: findかどっちかに統一する
    concat = S.finalizers.concat
    one = S.finalizers.one
    one_or = S.finalizers.one_or
    one_or_raise = S.finalizers.one_or_raise
    first = S.finalizers.first
    first_or = S.finalizers.first_or
    first_or_raise = S.finalizers.first_or_raise
    last = S.finalizers.last
    last_or = S.finalizers.last_or
    last_or_raise = S.finalizers.last_or_raise
    each = S.finalizers.each
    each_unpack = S.finalizers.each_unpack
    each_async = S.finalizers.each_async
    each_async_unpack = S.finalizers.each_async_unpack
    dispatch = S.concurrent.dispatch
    to = S.finalizers.to
    lazy = Lazy


class AsyncFinalizerBase:
    empty = A.finalizers.empty
    exists = A.finalizers.exists
    len = A.finalizers._len
    all = A.finalizers._all
    any = A.finalizers._any
    sum = A.finalizers._sum
    min = A.finalizers._min
    max = A.finalizers._max
    average = A.finalizers.average
    reduce = A.finalizers.reduce
    accumulate = A.finalizers.accumulate
    find = A.finalizers.find
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
    each_unpack = A.finalizers.each_unpack
    each_async = A.finalizers.each_async
    each_async_unpack = A.finalizers.each_async_unpack
    dispatch = A.concurrent.dispatch
    to = A.finalizers.to
    lazy = Lazy


class Finalizer(FinalizerBase, Iterable[T]):
    def __init__(self, source: Iterable[T]):
        self.source = source

    def __iter__(self):
        return self.source.__iter__()

    def result(self, timeout: float = None) -> List[T]:
        return list(self.source)


class AsyncFinalizer(AsyncFinalizerBase, AsyncIterable[T]):
    def __init__(self, source: AsyncIterable[T]):
        self.source = source

    def __aiter__(self):
        return self.source.__aiter__()

    def __await__(self):
        return self._result_async().__await__()

    async def _result_async(self):
        return [x async for x in self]
