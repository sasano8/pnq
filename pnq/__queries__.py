from typing import (
    TYPE_CHECKING,
    Any,
    AsyncGenerator,
    AsyncIterable,
    AsyncIterator,
    Awaitable,
    Callable,
    Dict,
    FrozenSet,
    Generator,
    Generic,
    Iterable,
    Iterator,
    List,
    Mapping,
    NoReturn,
    Sequence,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
    no_type_check,
    overload,
)

try:
    from typing import Literal
except:
    from typing_extensions import Literal

from pnq.exceptions import NotFoundError

from ._itertools import builder, core, queryables
from ._itertools.op import TH_ASSIGN_OP
from ._itertools.queryables import AsyncFinalizer, Finalizer
from ._itertools.requests import Response

T = TypeVar("T")
K = TypeVar("K")
V = TypeVar("V")
K2 = TypeVar("K2")
V2 = TypeVar("V2")
R = TypeVar("R")


__all__ = ["Query", "PairQuery", "query"]


{% for query in queries %}

class {{query.CLS}}:
    if TYPE_CHECKING:
        def __iter__(self) -> Iterator[{{query.T}}]:
            ...

        def __aiter__(self) -> AsyncIterator[{{query.T}}]:
            ...

    def as_iter(self) -> "Finalizer[{{query.T}}]":
        # return finalizers.SyncFinalizer(self)
        return Finalizer(self)

    def as_aiter(self) -> "Finalizer[{{query.T}}]":
        # return finalizers.AsyncFinalizer(self)
        return AsyncFinalizer(self)

    @property
    def _(self) -> "Finalizer[{{query.T}}]":
        # TODO: queryables.Finalizerを使うこと
        # return finalizers.AsyncFinalizer(self)
        return AsyncFinalizer(self)


    async def _call(self):
        return await PnqList.from_aiter(self)
        # return QuerySeq([x async for x in self])

    {% if query.is_pair %}

    def save(self) -> "PnqListPair[{{query.K}}, {{query.V}}]":
        return PnqList(self)

    result = save

    def __await__(self) -> Generator[Any, Any, "PnqListPair[{{query.K}}, {{query.V}}]"]:
        return self._call().__await__()

    {% else %}

    def save(self) -> "PnqList[{{query.T}}]":
        return PnqList(self)

    result = save

    def __await__(self) -> Generator[Any, Any, "PnqList[{{query.T}}]"]:
        return self._call().__await__()

    {% endif %}

    def len(self) -> int:
        return Finalizer.len(self)

    def exists(self) -> bool:
        return Finalizer.exists(self)

    def all(self, selector: Callable[[{{query.T}}], Any]=lambda x: x) -> bool:
        return Finalizer.all(self, selector)

    def any(self, selector: Callable[[{{query.T}}], Any]=lambda x: x) -> bool:
        return Finalizer.any(self, selector)

    def contains(self, value, selector: Callable[[{{query.T}}], Any]=lambda x: x) -> bool:
        return Finalizer.contains(self, value, selector)

    @overload
    def min(self, *, default=NoReturn) -> Union[{{query.T}}, NoReturn]: ...

    @overload
    def min(self, key_selector: Callable[[{{query.T}}], R]=lambda x: x, default=NoReturn) -> R: ...
    def min(self, key_selector: Callable[[{{query.T}}], R]=lambda x: x, default=NoReturn) -> R:
        return Finalizer.min(self, key_selector, default)

    @overload
    def max(self, *, default=NoReturn) -> Union[{{query.T}}, NoReturn]: ...
    @overload
    def max(self, key_selector: Callable[[{{query.T}}], R]=lambda x: x, default=NoReturn) -> R: ...
    def max(self, key_selector: Callable[[{{query.T}}], R]=lambda x: x, default=NoReturn) -> R:
        return Finalizer.max(self, key_selector, default)

    @overload
    def sum(self) -> {{query.T}}: ...
    @overload
    def sum(self, selector: Callable[[{{query.T}}], R]=lambda x: x) -> R: ...
    def sum(self, selector: Callable[[{{query.T}}], R]=lambda x: x) -> R:
        return Finalizer.sum(self, selector)

    @overload
    def average(self) -> {{query.T}}: ...
    @overload
    def average(self, selector: Callable[[{{query.T}}], R]=lambda x: x) -> R: ...
    def average(self, selector: Callable[[{{query.T}}], R]=lambda x: x) -> R:
        return Finalizer.average(self, selector)

    def reduce(self, seed: T, op: Union[TH_ASSIGN_OP, Callable[[Any, Any], Any]] = "+=", selector=lambda x: x) -> T:
        return Finalizer.reduce(self, seed, op, selector)

    def concat(self, selector=lambda x: x, delimiter: str = "") -> str:
        return Finalizer.concat(self, selector, delimiter)

    def test(self, expect):
        """同期イテレータと非同期イテレータを実行し結果を比較し、
        それぞれの比較結果(bool)をタプルとして返す。
        実行不可な場合はNoneを返す。
        """

        # TODO: asyncを通るようにパスを切り替えるクエリを挟む必要がある

        import asyncio

        async def to_list():
            return [x async for x in self]

        sync_result = None
        async_result = None

        try:
            sync_result = [x for x in self]
        except Exception as e:
            pass

        try:
            async_result = asyncio.run(to_list())
        except Exception as e:
            pass

        return_sync = None
        return_async = None

        if sync_result is not None:
            return_sync = (sync_result == expect)

        if async_result is not None:
            return_async = (async_result == expect)

        return return_sync, return_async

    {% if query.is_pair %}

    @overload
    def to(self: Iterable[Tuple[K, V]], func: Type[Mapping[K, V]]) -> Mapping[K, V]:
        ...

    @overload
    def to(self: Iterable[Tuple[K, V]], func: Callable[[Iterable[Tuple[K, V]]], R]) -> R:
        ...

    {% endif %}

    @overload
    def to(self: Iterable[T], func: Type[Iterable[T]]) -> Iterable[T]:
        ...

    @overload
    def to(self: Iterable[T], func: Callable[[Iterable[T]], R]) -> R:
        ...

    def to(self, func: Callable[[Iterable[T]], R]) -> R:
        return Finalizer.to(self, func)

    def lazy(self, func, *args, **kwargs):
        return Finalizer.lazy(self, func, *args, **kwargs)

    def each(self, func: Callable = lambda x: x):
        return Finalizer.each(self, func)

    def each_unpack(self, func: Callable = lambda x: x):
        return Finalizer.each_unpack(self, func)

    async def each_async(self, func: Callable = lambda x: x):
        return await Finalizer.each_async(self, func)

    async def each_async_unpack(self, func: Callable = lambda x: x):
        return await Finalizer.each_async_unpack(self, func)

    def dispatch(self, func, executor: "PExecutor", *, unpack="", chunksize=1, callback=None):
        return Finalizer.dispatch(self, func, executor, unpack=unpack, chunksize=chunksize, callback=callback)

    def one(self) -> {{query.T}}:
        return Finalizer.one(self)

    def one_or(self, default: R) -> Union[{{query.T}}, R]:
        return Finalizer.one_or(self, default)

    def one_or_raise(self, exc: Union[str, Exception]) -> {{query.T}}:
        return Finalizer.one_or_raise(self, exc)

    def first(self) -> {{query.T}}:
        return Finalizer.first(self)

    def first_or(self, default: R) -> Union[{{query.T}}, R]:
        return Finalizer.first_or(self, default)

    def first_or_raise(self, exc: Union[str, Exception]) -> {{query.T}}:
        return Finalizer.first_or_raise(self, exc)

    def last(self) -> {{query.T}}:
        return Finalizer.last(self)

    def last_or(self, default: R) -> Union[{{query.T}}, R]:
        return Finalizer.last_or(self, default)

    def last_or_raise(self, exc: Union[str, Exception]) -> {{query.T}}:
        return Finalizer.last_or_raise(self, exc)

    @overload
    def cast(self, type: Type[Tuple[K2, V2]]) -> "{{pair.SELF__}}[K2, V2]":
        pass

    @overload
    def cast(self, type: Type[R]) -> "Query[R]":
        pass

    def cast(self, type: Type[R]) -> "Query[R]":
        return self

    def enumerate(self, start: int = 0, step: int = 1) -> "{{pair.SELF__}}[int, {{query.T}}]":
        return queryables.Enumerate(self, start, step)

    @overload
    def map(self, selector: Callable[[{{query.T}}], Tuple[K2, V2]]) -> "{{pair.SELF__}}[K2, V2]":
        pass

    @overload
    def map(self, selector: Callable[[{{query.T}}], R]) -> "{{sequence.SELF__}}[R]":
        pass

    def map(self, selector):
        return queryables.MapNullable(self, selector)

    {% if query.is_pair %}

    @overload
    def select(self, item: Literal[0]) -> "Query[{{query.K}}]":
        ...

    @overload
    def select(self, item: Literal[1]) -> "Query[{{query.V}}]":
        ...

    {% endif %}

    @overload
    def select(self, field) -> "Query[Any]":
        ...

    @overload
    def select(self, field1, field2, *fields) -> "PairQuery":
        ...

    @overload
    def select(self, field, *fields) -> "Query[Tuple]":
        ...

    def select(self, *fields, attr: bool = False) -> "Query[Any]":
        return queryables.Select(self, *fields, attr=attr)

    def select_as_tuple(self, *fields, attr: bool = False) -> "Query[Tuple]":
        return queryables.SelectAsTuple(self, *fields, attr=attr)

    def select_as_dict(self, *fields, attr: bool = False, default=NoReturn) -> "Query[Dict]":
        return queryables.SelectAsDict(self, *fields, attr=attr, default=default)

    def reflect(self, mapping, default=NoReturn, attr: bool = False):
        return queryables.Reflect(self, mapping, attr=attr)

    def flat(self, selector: Callable[..., Iterable[R]] = None) -> "{{sequence.SELF__}}[R]":
        return queryables.Flat(self, selector)

    def flat_recursive(self, selector: Callable[[{{query.T}}], Iterable[{{query.T}}]]) -> "{{sequence.SELF__}}[{{query.T}}]":
        return queryables.FlatRecursive(self, selector)

    def unpack_pos(self, selector: Callable[..., R]) -> "{{sequence.SELF__}}[R]":
        return queryables.UnpackPos(self, selector=selector)

    def unpack_kw(self, selector: Callable[..., R]) -> "{{sequence.SELF__}}[R]":
        return queryables.UnpackKw(self, selector=selector)

    def pivot_unstack(self, default=None) -> "PairQuery[Any, List]":
        return queryables.PivotUnstack(self, default=default)

    def pivot_stack(self) -> "Query[Dict]":
        return queryables.PivotStack(self)

    def group_by(self, selector: Callable[[{{query.T}}], Tuple[K2, V2]] = lambda x: x) -> "{{pair.SELF__}}[K2, List[V2]]":
        return queryables.GroupBy(self, selector=selector)

    def chunked(self, size: int) -> "Query[List[{{query.T}}]]":
        return queryables.Chunked(self, size=size)

    def tee(self, size: int):
        return queryables.Tee(self, size=size)

    def join(self, right, on: Callable[[Tuple[list, list]], Callable], select):
        return queryables.Join(self, right, on=on, select=select)

    def request(self, func, retry: int = None) -> "Query[Response]":
        return queryables.Request(self, func, retry)

    def request_async(self, func, retry: int = None, timeout=None) -> "Query[Response]":
        return queryables.RequestAsync(self, func, retry=retry, timeout=None)

    @overload
    def parallel(self, func: Callable[..., Awaitable[R]], executor=None, *, unpack="", chunksize=1) -> "Query[R]":
        ...

    @overload
    def parallel(self, func: Callable[..., R], executor=None, *, unpack="", chunksize=1) -> "Query[R]":
        ...

    def parallel(self, func, executor=None, *, unpack="", chunksize=1) -> "Query[R]":
        return queryables.Parallel(self, func, executor, unpack=unpack, chunksize=chunksize)

    def debug(self, breakpoint=lambda x: x, printer=print) -> "{{query.SELF_T}}":
        return queryables.Debug(self, breakpoint=breakpoint, printer=printer)

    def debug_path(self, selector_sync=lambda x: -10, selector_async=lambda x: 10) -> "{{query.SELF_T}}":
        return queryables.DebugPath(self, selector_sync, selector_async)

    def filter(self, predicate: Callable[[{{query.T}}], bool]) -> "{{query.SELF_T}}":
        return queryables.Filter(self, predicate)

    def filter_type(self, *types: Type[R]) -> "{{query.SELF_T}}":
        return queryables.FilterType(self, *types)

    @overload
    def filter_unique(self) -> "{{query.SELF_T}}":
        ...

    @overload
    def filter_unique(self, selector: Callable[[{{query.T}}], Tuple[K2, V2]]) -> "{{pair.SELF__}}[K2, V2]":
        ...

    @overload
    def filter_unique(self, selector: Callable[[{{query.T}}], R]) -> "{{sequence.SELF__}}[R]":
        ...

    def filter_unique(self, selector=None):
        return queryables.FilterUnique(self, selector=selector)

    def distinct(self, selector: Callable[[{{query.T}}], Any]) -> "{{query.SELF_T}}":
        return queryables.FilterUnique(self, selector=selector)

    def must(self, predicate: Callable[[{{query.T}}], bool], msg: str="") -> "{{query.SELF_T}}":
        return queryables.Must(self, predicate, msg)

    def must_type(self, type, *types: Type) -> "{{query.SELF_T}}":
        return queryables.MustType(self, type, *types)

    def must_unique(self, selector: Callable[[T], R] = None):
        return queryables.MustUnique(self, selector=selector)

    def take(self, count_or_range: Union[int, range]) -> "{{query.SELF_T}}":
        return queryables.Take(self, count_or_range)

    def take_while(self, predicate) -> "{{query.SELF_T}}":
        return queryables.TakeWhile(self, predicate)

    def skip(self, count_or_range: Union[int, range]) -> "{{query.SELF_T}}":
        return queryables.Skip(self, count_or_range)

    def take_page(self, page: int, size: int) -> "{{query.SELF_T}}":
        return queryables.TakePage(self, page=page, size=size)

    def order_by(self, *fields, desc: bool = False, attr: bool = False) -> "{{query.SELF_T}}":
        return queryables.OrderBy(self, *fields, desc=desc, attr=attr)

    def order_by_map(self, selector=None, *, desc: bool = False) -> "{{query.SELF_T}}":
        return queryables.OrderByMap(self, selector=selector, desc=desc)

    def order_by_reverse(self) -> "{{query.SELF_T}}":
        return queryables.OrderByReverse(self)

    def order_by_shuffle(self) -> "{{query.SELF_T}}":
        return queryables.OrderByShuffle(self)

    def sleep(self, seconds: float) -> "{{query.SELF_T}}":
        return queryables.Sleep(self, seconds)

    def zip(self):
        return queryables.Zip(self)

    def cartesian(self, *iterables) -> "Query[Tuple]":
        return queryables.Cartesian(self, *iterables)

{% endfor %}


if not TYPE_CHECKING:
    import types

    class Queries:
        pass

    from ._itertools import queryables

    classess = Queries()

    for cls in queryables.exports:
        baseclasses = (cls, Query[T])
        created = types.new_class(cls.__name__, baseclasses)

        setattr(classess, cls.__name__, created)

    queryables = classess


class QueryBase(Query[T], core.Query[T]):
    pass


class QueryAsync(Query[T], core.QueryAsync[T]):
    pass


class QueryNormal(Query[T], core.QueryNormal[T]):
    pass


class QueryDict(PairQuery[K, V], core.QueryDict[K, V]):
    def filter_keys(self, *keys) -> "PairQuery[K, V]":
        return queryables.FilterKeys(self, *keys)

    # @no_type_check
    def must_keys(self, *keys) -> "PairQuery[K, V]":
        return queryables.MustKeys(self, *keys, typ="map")

    def get(self, key) -> V:
        try:
            return self.source[key]
        except KeyError:
            raise NotFoundError(key)

    def get_or(self, key, default: R = None) -> Union[V, R]:
        try:
            return self.get(key)
        except NotFoundError:
            return default

    def get_or_raise(self, key, exc: Union[str, Exception]) -> V:
        undefined = object()
        result = self.get_or(key, undefined)
        if result is undefined:
            if isinstance(exc, str):
                raise Exception(exc)
            else:
                raise exc
        else:
            return result


class QuerySeq(Query[T], core.QuerySeq[T]):
    def filter_keys(self, *keys) -> "Query[T]":
        return queryables.FilterKeys(self, *keys)

    def must_keys(self, *keys) -> "Query[T]":
        return queryables.MustKeys(self, *keys, typ="seq")

    def get(self, key) -> T:
        try:
            return self.source[key]
        except IndexError:
            raise NotFoundError(key)

    def get_or(self, key, default: R = None) -> Union[T, R]:
        try:
            return self.get(key)
        except NotFoundError:
            return default

    def get_or_raise(self, key, exc: Union[str, Exception]) -> T:
        undefined = object()
        result = self.get_or(key, undefined)
        if result is undefined:
            if isinstance(exc, str):
                raise Exception(exc)
            else:
                raise exc
        else:
            return result


class PnqList(Query[T], List[T]):
    @property
    def source(self):
        return self

    @classmethod
    async def from_aiter(cls, aiter: AsyncIterable[T]) -> "PnqList[T]":
        result = cls()
        async for x in aiter:
            result.append(x)

        return result

    def filter_keys(self, *keys) -> "Query[T]":
        return queryables.FilterKeys(self, *keys)

    def must_keys(self, *keys) -> "Query[T]":
        return queryables.MustKeys(self, *keys, typ="seq")

    def get(self, key) -> T:
        try:
            return self.source[key]
        except IndexError:
            raise NotFoundError(key)

    def get_or(self, key, default: R = None) -> Union[T, R]:
        try:
            return self.get(key)
        except NotFoundError:
            return default

    def get_or_raise(self, key, exc: Union[str, Exception]) -> T:
        undefined = object()
        result = self.get_or(key, undefined)
        if result is undefined:
            if isinstance(exc, str):
                raise Exception(exc)
            else:
                raise exc
        else:
            return result


class QuerySet(Query[T], core.QuerySet[T]):
    def filter_keys(self, *keys) -> "Query[T]":
        return queryables.FilterKeys(self, *keys)

    def must_keys(self, *keys) -> "Query[T]":
        return queryables.MustKeys(self, *keys, typ="set")

    def get(self, key) -> T:
        if key in self.source:
            return key
        else:
            raise NotFoundError(key)

    def get_or(self, key, default: R = None) -> Union[T, R]:
        try:
            return self.get(key)
        except NotFoundError:
            return default

    def get_or_raise(self, key, exc: Union[str, Exception]) -> T:
        undefined = object()
        result = self.get_or(key, undefined)
        if result is undefined:
            if isinstance(exc, str):
                raise Exception(exc)
            else:
                raise exc
        else:
            return result

if TYPE_CHECKING:
    class QuerySeqPair(PairQuery[K, V], core.QuerySeq[Tuple[K, V]]):
        ...

    class QuerySetPair(PairQuery[K, V], core.QuerySeq[Tuple[K, V]]):
        ...

    class PnqListPair(PairQuery[K, V], PnqList[Tuple[K, V]]):
        def __init__(self, source: Iterable[Tuple[K, V]]):
            ...


class QueryBuilder(builder.Builder):
    QUERY_BOTH = QueryBase
    QUERY_ASYNC = QueryAsync
    QUERY_NORMAL = QueryNormal
    QUERY_SEQ = QuerySeq
    QUERY_DICT = QueryDict
    QUERY_SET = QuerySet


@overload
def query(source: Mapping[K, V]) -> QueryDict[K, V]:
    ...


@overload
def query(source: Set[Tuple[K, V]]) -> "QuerySetPair[K, V]":
    ...

@overload
def query(source: Set[T]) -> QuerySet[T]:
    ...

@overload
def query(source: Iterable[Tuple[K, V]]) -> "QuerySeqPair[K, V]":
    ...


@overload
def query(source: AsyncIterable[Tuple[K, V]]) -> "QuerySeqPair[K, V]":
    ...

@overload
def query(source: Generator[T, Any, Any]) -> QuerySeq[T]:
    ...

@overload
def query(source: AsyncGenerator[T, Any]) -> QuerySeq[T]:
    ...

@overload
def query(source: Iterable[T]) -> QuerySeq[T]:
    ...

@overload
def query(source: AsyncIterable[T]) -> QuerySeq[T]:
    ...

@overload
def query(source: Iterable[T]) -> QuerySeq[T]:
    ...

@overload
def query(source) -> "Query[Any]":
    ...


def query(source):
    return QueryBuilder.query(source)


run = QueryBuilder.run
