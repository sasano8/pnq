import sys
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

from pnq.exceptions import NoElementError, NotFoundError, NotOneElementError

from ._itertools import actions, builder, core, finalizers
from ._itertools import queryables as queries
from ._itertools.op import TH_ASSIGN_OP
from ._itertools.requests import Response

# from . import actions, core, builder, queries
# from .base.exceptions import NoElementError, NotFoundError, NotOneElementError
# from .base.op import TH_ASSIGN_OP
# from .base.requests import Response

# from .base import finalizers


T = TypeVar("T")
K = TypeVar("K")
V = TypeVar("V")
K2 = TypeVar("K2")
V2 = TypeVar("V2")
R = TypeVar("R")


__all__ = ["Query", "PairQuery", "query"]


class Query(Generic[T]):
    if TYPE_CHECKING:

        def __iter__(self) -> Iterator[T]:
            ...

        def __aiter__(self) -> AsyncIterator[T]:
            ...

    def as_iter(self) -> "finalizers.SyncFinalizer[T]":
        return finalizers.SyncFinalizer(self)

    def as_aiter(self) -> "finalizers.AsyncFinalizer[T]":
        return finalizers.AsyncFinalizer(self)

    @property
    def _(self) -> "finalizers.AsyncFinalizer[T]":
        return finalizers.AsyncFinalizer(self)

    async def _call(self):
        return await PnqList.from_aiter(self)
        # return QuerySeq([x async for x in self])

    def save(self) -> "PnqList[T]":
        return PnqList(self)

    result = save

    def __await__(self) -> Generator[Any, Any, "PnqList[T]"]:
        return self._call().__await__()

    def len(self) -> int:
        return actions.len(self)

    def exists(self) -> bool:
        return actions.exists(self)

    def all(self, selector: Callable[[T], Any] = lambda x: x) -> bool:
        return actions.all(self, selector)

    def any(self, selector: Callable[[T], Any] = lambda x: x) -> bool:
        return actions.any(self, selector)

    def contains(self, value, selector: Callable[[T], Any] = lambda x: x) -> bool:
        return actions.contains(self, value, selector)

    @overload
    def min(self, *, default=NoReturn) -> Union[T, NoReturn]:
        ...

    @overload
    def min(self, selector: Callable[[T], R] = lambda x: x, default=NoReturn) -> R:
        ...

    def min(self, selector: Callable[[T], R] = lambda x: x, default=NoReturn) -> R:
        return actions.min(self, selector, default)

    @overload
    def max(self, *, default=NoReturn) -> Union[T, NoReturn]:
        ...

    @overload
    def max(self, selector: Callable[[T], R] = lambda x: x, default=NoReturn) -> R:
        ...

    def max(self, selector: Callable[[T], R] = lambda x: x, default=NoReturn) -> R:
        return actions.max(self, selector, default)

    @overload
    def sum(self) -> T:
        ...

    @overload
    def sum(self, selector: Callable[[T], R] = lambda x: x) -> R:
        ...

    def sum(self, selector: Callable[[T], R] = lambda x: x) -> R:
        return actions.sum(self, selector)

    @overload
    def average(self) -> T:
        ...

    @overload
    def average(self, selector: Callable[[T], R] = lambda x: x) -> R:
        ...

    def average(self, selector: Callable[[T], R] = lambda x: x) -> R:
        return actions.average(self, selector)

    def reduce(
        self,
        seed: T,
        op: Union[TH_ASSIGN_OP, Callable[[Any, Any], Any]] = "+=",
        selector=lambda x: x,
    ) -> T:
        return actions.reduce(self, seed, op, selector)

    def concat(self, selector=lambda x: x, delimiter: str = "") -> str:
        return actions.concat(self, selector, delimiter)

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
            return_sync = sync_result == expect

        if async_result is not None:
            return_async = async_result == expect

        return return_sync, return_async

    @overload
    def to(self: Iterable[T], func: Type[Iterable[T]]) -> Iterable[T]:
        ...

    @overload
    def to(self: Iterable[T], func: Callable[[Iterable[T]], R]) -> R:
        ...

    def to(self, func: Callable[[Iterable[T]], R]) -> R:
        return actions.to(self, func)

    def lazy(self, func, *args, **kwargs):
        return queries.Lazy(self, func, *args, **kwargs)

    def each(self, func: Callable = lambda x: x):
        return actions.each(self, func)

    def each_unpack(self, func: Callable = lambda x: x):
        return actions.each_unpack(self, func)

    async def each_async(self, func: Callable = lambda x: x):
        return await actions.each_async(self, func)

    async def each_async_unpack(self, func: Callable = lambda x: x):
        return await actions.each_async_unpack(self, func)

    def one(self) -> T:
        return actions.one(self)

    def one_or(self, default: R) -> Union[T, R]:
        return actions.one_or(self, default)

    def one_or_raise(self, exc: Union[str, Exception]) -> T:
        return actions.one_or_raise(self, exc)

    def first(self) -> T:
        return actions.first(self)

    def first_or(self, default: R) -> Union[T, R]:
        return actions.first_or(self, default)

    def first_or_raise(self, exc: Union[str, Exception]) -> T:
        return actions.first_or_raise(self, exc)

    def last(self) -> T:
        return actions.last(self)

    def last_or(self, default: R) -> Union[T, R]:
        return actions.last_or(self, default)

    def last_or_raise(self, exc: Union[str, Exception]) -> T:
        return actions.last_or_raise(self, exc)

    @overload
    def cast(self, type: Type[Tuple[K2, V2]]) -> "PairQuery[K2, V2]":
        pass

    @overload
    def cast(self, type: Type[R]) -> "Query[R]":
        pass

    def cast(self, type: Type[R]) -> "Query[R]":
        return self

    def enumerate(self, start: int = 0, step: int = 1) -> "PairQuery[int, T]":
        return queries.Enumerate(self, start, step)

    @overload
    def map(self, selector: Callable[[T], Tuple[K2, V2]]) -> "PairQuery[K2, V2]":
        pass

    @overload
    def map(self, selector: Callable[[T], R]) -> "Query[R]":
        pass

    def map(self, selector):
        return queries.MapNullable(self, selector)

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
        return queries.Select(self, *fields, attr=attr)

    def select_as_tuple(self, *fields, attr: bool = False) -> "Query[Tuple]":
        return queries.SelectAsTuple(self, *fields, attr=attr)

    def select_as_dict(
        self, *fields, attr: bool = False, default=NoReturn
    ) -> "Query[Dict]":
        return queries.SelectAsDict(self, *fields, attr=attr, default=default)

    def reflect(self, mapping, default=NoReturn, attr: bool = False):
        return queries.Reflect(self, mapping, attr=attr)

    def flat(self, selector: Callable[..., Iterable[R]] = None) -> "Query[R]":
        return queries.Flat(self, selector)

    def flat_recursive(self, selector: Callable[[T], Iterable[T]]) -> "Query[T]":
        return queries.FlatRecursive(self, selector)

    def unpack_pos(self, selector: Callable[..., R]) -> "Query[R]":
        return queries.UnpackPos(self, selector=selector)

    def unpack_kw(self, selector: Callable[..., R]) -> "Query[R]":
        return queries.UnpackKw(self, selector=selector)

    def pivot_unstack(self, default=None) -> "PairQuery[Any, List]":
        return queries.PivotUnstack(self, default=default)

    def pivot_stack(self) -> "Query[Dict]":
        return queries.PivotStack(self)

    def group_by(
        self, selector: Callable[[T], Tuple[K2, V2]] = lambda x: x
    ) -> "PairQuery[K2, List[V2]]":
        return queries.GroupBy(self, selector=selector)

    def chunked(self, size: int) -> "Query[List[T]]":
        return queries.Chunked(self, size=size)

    def tee(self, size: int):
        return queries.Tee(self, size=size)

    def join(self, right, on: Callable[[Tuple[list, list]], Callable], select):
        [].join(
            [],
            lambda left, right: left.name == right.name,
            lambda left, right: (left.name, right.age),
        )

        table(User).join(Item, on=User.id == Item.id).select(User.id, Item.id)

        pass

    def request(self, func, retry: int = None) -> "Query[Response]":
        return queries.Request(self, func, retry)

    def request_async(self, func, retry: int = None, timeout=None) -> "Query[Response]":
        return queries.RequestAsync(self, func, retry=retry, timeout=None)

    @overload
    def parallel(
        self,
        func: Callable[..., Awaitable[R]],
        executor=None,
        *,
        unpack="",
        chunksize=1
    ) -> "Query[R]":
        ...

    @overload
    def parallel(
        self, func: Callable[..., R], executor=None, *, unpack="", chunksize=1
    ) -> "Query[R]":
        ...

    def parallel(self, func, executor=None, *, unpack="", chunksize=1) -> "Query[R]":
        return queries.Parallel(
            self, func, executor, unpack=unpack, chunksize=chunksize
        )

    def debug(self, breakpoint=lambda x: x, printer=print) -> "Query[T]":
        return queries.Debug(self, breakpoint=breakpoint, printer=printer)

    def debug_path(
        self, selector_sync=lambda x: -10, selector_async=lambda x: 10
    ) -> "Query[T]":
        return queries.DebugPath(self, selector_sync, selector_async)

    def filter(self, predicate: Callable[[T], bool]) -> "Query[T]":
        return queries.Filter(self, predicate)

    def filter_type(self, *types: Type[R]) -> "Query[T]":
        return queries.FilterType(self, *types)

    @overload
    def filter_unique(self) -> "Query[T]":
        ...

    @overload
    def filter_unique(
        self, selector: Callable[[T], Tuple[K2, V2]]
    ) -> "PairQuery[K2, V2]":
        ...

    @overload
    def filter_unique(self, selector: Callable[[T], R]) -> "Query[R]":
        ...

    def filter_unique(self, selector=None):
        return queries.FilterUnique(self, selector=selector)

    def distinct(self, selector: Callable[[T], Any]) -> "Query[T]":
        return queries.FilterUnique(self, selector=selector)

    def must(self, predicate: Callable[[T], bool], msg: str = "") -> "Query[T]":
        return queries.Must(self, predicate, msg)

    def must_type(self, type, *types: Type) -> "Query[T]":
        return queries.MustType(self, type, *types)

    def must_unique(self, selector: Callable[[T], R] = None):
        return queries.MustUnique(self, selector=selector)

    def take(self, count_or_range: Union[int, range]) -> "Query[T]":
        return queries.Take(self, count_or_range)

    def take_while(self, predicate) -> "Query[T]":
        return queries.TakeWhile(self, predicate)

    def skip(self, count_or_range: Union[int, range]) -> "Query[T]":
        return queries.Skip(self, count_or_range)

    def take_page(self, page: int, size: int) -> "Query[T]":
        return queries.TakePage(self, page=page, size=size)

    def order_by(self, *fields, desc: bool = False, attr: bool = False) -> "Query[T]":
        return queries.OrderBy(self, *fields, desc=desc, attr=attr)

    def order_by_map(self, selector=None, *, desc: bool = False) -> "Query[T]":
        return queries.OrderByMap(self, selector=selector, desc=desc)

    def order_by_reverse(self) -> "Query[T]":
        return queries.OrderByReverse(self)

    def order_by_shuffle(self) -> "Query[T]":
        return queries.OrderByShuffle(self)

    def sleep(self, seconds: float) -> "Query[T]":
        return queries.Sleep(self, seconds)

    # def sleep_async(self, seconds: float) -> "Query[T]":
    #     return queries.Sleep(self, seconds)

    def zip(self):
        raise NotImplementedError()

    def cartesian(self, *iterables) -> "Query[Tuple]":
        return queries.Cartesian(self, *iterables)


class PairQuery(Generic[K, V], Query[Tuple[K, V]]):
    if TYPE_CHECKING:

        def __iter__(self) -> Iterator[Tuple[K, V]]:
            ...

        def __aiter__(self) -> AsyncIterator[Tuple[K, V]]:
            ...

    def as_iter(self) -> "finalizers.SyncFinalizer[Tuple[K,V]]":
        return finalizers.SyncFinalizer(self)

    def as_aiter(self) -> "finalizers.AsyncFinalizer[Tuple[K,V]]":
        return finalizers.AsyncFinalizer(self)

    @property
    def _(self) -> "finalizers.AsyncFinalizer[Tuple[K,V]]":
        return finalizers.AsyncFinalizer(self)

    async def _call(self):
        return await PnqList.from_aiter(self)
        # return QuerySeq([x async for x in self])

    def save(self) -> "PnqListPair[K, V]":
        return PnqList(self)

    result = save

    def __await__(self) -> Generator[Any, Any, "PnqListPair[K, V]"]:
        return self._call().__await__()

    def len(self) -> int:
        return actions.len(self)

    def exists(self) -> bool:
        return actions.exists(self)

    def all(self, selector: Callable[[Tuple[K, V]], Any] = lambda x: x) -> bool:
        return actions.all(self, selector)

    def any(self, selector: Callable[[Tuple[K, V]], Any] = lambda x: x) -> bool:
        return actions.any(self, selector)

    def contains(
        self, value, selector: Callable[[Tuple[K, V]], Any] = lambda x: x
    ) -> bool:
        return actions.contains(self, value, selector)

    @overload
    def min(self, *, default=NoReturn) -> Union[Tuple[K, V], NoReturn]:
        ...

    @overload
    def min(
        self, selector: Callable[[Tuple[K, V]], R] = lambda x: x, default=NoReturn
    ) -> R:
        ...

    def min(
        self, selector: Callable[[Tuple[K, V]], R] = lambda x: x, default=NoReturn
    ) -> R:
        return actions.min(self, selector, default)

    @overload
    def max(self, *, default=NoReturn) -> Union[Tuple[K, V], NoReturn]:
        ...

    @overload
    def max(
        self, selector: Callable[[Tuple[K, V]], R] = lambda x: x, default=NoReturn
    ) -> R:
        ...

    def max(
        self, selector: Callable[[Tuple[K, V]], R] = lambda x: x, default=NoReturn
    ) -> R:
        return actions.max(self, selector, default)

    @overload
    def sum(self) -> Tuple[K, V]:
        ...

    @overload
    def sum(self, selector: Callable[[Tuple[K, V]], R] = lambda x: x) -> R:
        ...

    def sum(self, selector: Callable[[Tuple[K, V]], R] = lambda x: x) -> R:
        return actions.sum(self, selector)

    @overload
    def average(self) -> Tuple[K, V]:
        ...

    @overload
    def average(self, selector: Callable[[Tuple[K, V]], R] = lambda x: x) -> R:
        ...

    def average(self, selector: Callable[[Tuple[K, V]], R] = lambda x: x) -> R:
        return actions.average(self, selector)

    def reduce(
        self,
        seed: T,
        op: Union[TH_ASSIGN_OP, Callable[[Any, Any], Any]] = "+=",
        selector=lambda x: x,
    ) -> T:
        return actions.reduce(self, seed, op, selector)

    def concat(self, selector=lambda x: x, delimiter: str = "") -> str:
        return actions.concat(self, selector, delimiter)

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
            return_sync = sync_result == expect

        if async_result is not None:
            return_async = async_result == expect

        return return_sync, return_async

    @overload
    def to(self: Iterable[Tuple[K, V]], func: Type[Mapping[K, V]]) -> Mapping[K, V]:
        ...

    @overload
    def to(
        self: Iterable[Tuple[K, V]], func: Callable[[Iterable[Tuple[K, V]]], R]
    ) -> R:
        ...

    @overload
    def to(self: Iterable[T], func: Type[Iterable[T]]) -> Iterable[T]:
        ...

    @overload
    def to(self: Iterable[T], func: Callable[[Iterable[T]], R]) -> R:
        ...

    def to(self, func: Callable[[Iterable[T]], R]) -> R:
        return actions.to(self, func)

    def lazy(self, func, *args, **kwargs):
        return queries.Lazy(self, func, *args, **kwargs)

    def each(self, func: Callable = lambda x: x):
        return actions.each(self, func)

    def each_unpack(self, func: Callable = lambda x: x):
        return actions.each_unpack(self, func)

    async def each_async(self, func: Callable = lambda x: x):
        return await actions.each_async(self, func)

    async def each_async_unpack(self, func: Callable = lambda x: x):
        return await actions.each_async_unpack(self, func)

    def one(self) -> Tuple[K, V]:
        return actions.one(self)

    def one_or(self, default: R) -> Union[Tuple[K, V], R]:
        return actions.one_or(self, default)

    def one_or_raise(self, exc: Union[str, Exception]) -> Tuple[K, V]:
        return actions.one_or_raise(self, exc)

    def first(self) -> Tuple[K, V]:
        return actions.first(self)

    def first_or(self, default: R) -> Union[Tuple[K, V], R]:
        return actions.first_or(self, default)

    def first_or_raise(self, exc: Union[str, Exception]) -> Tuple[K, V]:
        return actions.first_or_raise(self, exc)

    def last(self) -> Tuple[K, V]:
        return actions.last(self)

    def last_or(self, default: R) -> Union[Tuple[K, V], R]:
        return actions.last_or(self, default)

    def last_or_raise(self, exc: Union[str, Exception]) -> Tuple[K, V]:
        return actions.last_or_raise(self, exc)

    @overload
    def cast(self, type: Type[Tuple[K2, V2]]) -> "PairQuery[K2, V2]":
        pass

    @overload
    def cast(self, type: Type[R]) -> "Query[R]":
        pass

    def cast(self, type: Type[R]) -> "Query[R]":
        return self

    def enumerate(self, start: int = 0, step: int = 1) -> "PairQuery[int, Tuple[K,V]]":
        return queries.Enumerate(self, start, step)

    @overload
    def map(
        self, selector: Callable[[Tuple[K, V]], Tuple[K2, V2]]
    ) -> "PairQuery[K2, V2]":
        pass

    @overload
    def map(self, selector: Callable[[Tuple[K, V]], R]) -> "Query[R]":
        pass

    def map(self, selector):
        return queries.MapNullable(self, selector)

    @overload
    def select(self, item: Literal[0]) -> "Query[K]":
        ...

    @overload
    def select(self, item: Literal[1]) -> "Query[V]":
        ...

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
        return queries.Select(self, *fields, attr=attr)

    def select_as_tuple(self, *fields, attr: bool = False) -> "Query[Tuple]":
        return queries.SelectAsTuple(self, *fields, attr=attr)

    def select_as_dict(
        self, *fields, attr: bool = False, default=NoReturn
    ) -> "Query[Dict]":
        return queries.SelectAsDict(self, *fields, attr=attr, default=default)

    def reflect(self, mapping, default=NoReturn, attr: bool = False):
        return queries.Reflect(self, mapping, attr=attr)

    def flat(self, selector: Callable[..., Iterable[R]] = None) -> "Query[R]":
        return queries.Flat(self, selector)

    def flat_recursive(
        self, selector: Callable[[Tuple[K, V]], Iterable[Tuple[K, V]]]
    ) -> "Query[Tuple[K,V]]":
        return queries.FlatRecursive(self, selector)

    def unpack_pos(self, selector: Callable[..., R]) -> "Query[R]":
        return queries.UnpackPos(self, selector=selector)

    def unpack_kw(self, selector: Callable[..., R]) -> "Query[R]":
        return queries.UnpackKw(self, selector=selector)

    def pivot_unstack(self, default=None) -> "PairQuery[Any, List]":
        return queries.PivotUnstack(self, default=default)

    def pivot_stack(self) -> "Query[Dict]":
        return queries.PivotStack(self)

    def group_by(
        self, selector: Callable[[Tuple[K, V]], Tuple[K2, V2]] = lambda x: x
    ) -> "PairQuery[K2, List[V2]]":
        return queries.GroupBy(self, selector=selector)

    def chunked(self, size: int) -> "Query[List[Tuple[K,V]]]":
        return queries.Chunked(self, size=size)

    def tee(self, size: int):
        return queries.Tee(self, size=size)

    def join(self, right, on: Callable[[Tuple[list, list]], Callable], select):
        [].join(
            [],
            lambda left, right: left.name == right.name,
            lambda left, right: (left.name, right.age),
        )

        table(User).join(Item, on=User.id == Item.id).select(User.id, Item.id)

        pass

    def request(self, func, retry: int = None) -> "Query[Response]":
        return queries.Request(self, func, retry)

    def request_async(self, func, retry: int = None, timeout=None) -> "Query[Response]":
        return queries.RequestAsync(self, func, retry=retry, timeout=None)

    @overload
    def parallel(
        self,
        func: Callable[..., Awaitable[R]],
        executor=None,
        *,
        unpack="",
        chunksize=1
    ) -> "Query[R]":
        ...

    @overload
    def parallel(
        self, func: Callable[..., R], executor=None, *, unpack="", chunksize=1
    ) -> "Query[R]":
        ...

    def parallel(self, func, executor=None, *, unpack="", chunksize=1) -> "Query[R]":
        return queries.Parallel(
            self, func, executor, unpack=unpack, chunksize=chunksize
        )

    def debug(self, breakpoint=lambda x: x, printer=print) -> "PairQuery[K,V]":
        return queries.Debug(self, breakpoint=breakpoint, printer=printer)

    def debug_path(
        self, selector_sync=lambda x: -10, selector_async=lambda x: 10
    ) -> "PairQuery[K,V]":
        return queries.DebugPath(self, selector_sync, selector_async)

    def filter(self, predicate: Callable[[Tuple[K, V]], bool]) -> "PairQuery[K,V]":
        return queries.Filter(self, predicate)

    def filter_type(self, *types: Type[R]) -> "PairQuery[K,V]":
        return queries.FilterType(self, *types)

    @overload
    def filter_unique(self) -> "PairQuery[K,V]":
        ...

    @overload
    def filter_unique(
        self, selector: Callable[[Tuple[K, V]], Tuple[K2, V2]]
    ) -> "PairQuery[K2, V2]":
        ...

    @overload
    def filter_unique(self, selector: Callable[[Tuple[K, V]], R]) -> "Query[R]":
        ...

    def filter_unique(self, selector=None):
        return queries.FilterUnique(self, selector=selector)

    def distinct(self, selector: Callable[[Tuple[K, V]], Any]) -> "PairQuery[K,V]":
        return queries.FilterUnique(self, selector=selector)

    def must(
        self, predicate: Callable[[Tuple[K, V]], bool], msg: str = ""
    ) -> "PairQuery[K,V]":
        return queries.Must(self, predicate, msg)

    def must_type(self, type, *types: Type) -> "PairQuery[K,V]":
        return queries.MustType(self, type, *types)

    def must_unique(self, selector: Callable[[T], R] = None):
        return queries.MustUnique(self, selector=selector)

    def take(self, count_or_range: Union[int, range]) -> "PairQuery[K,V]":
        return queries.Take(self, count_or_range)

    def take_while(self, predicate) -> "PairQuery[K,V]":
        return queries.TakeWhile(self, predicate)

    def skip(self, count_or_range: Union[int, range]) -> "PairQuery[K,V]":
        return queries.Skip(self, count_or_range)

    def take_page(self, page: int, size: int) -> "PairQuery[K,V]":
        return queries.TakePage(self, page=page, size=size)

    def order_by(
        self, *fields, desc: bool = False, attr: bool = False
    ) -> "PairQuery[K,V]":
        return queries.OrderBy(self, *fields, desc=desc, attr=attr)

    def order_by_map(self, selector=None, *, desc: bool = False) -> "PairQuery[K,V]":
        return queries.OrderByMap(self, selector=selector, desc=desc)

    def order_by_reverse(self) -> "PairQuery[K,V]":
        return queries.OrderByReverse(self)

    def order_by_shuffle(self) -> "PairQuery[K,V]":
        return queries.OrderByShuffle(self)

    def sleep(self, seconds: float) -> "PairQuery[K,V]":
        return queries.Sleep(self, seconds)

    # def sleep_async(self, seconds: float) -> "PairQuery[K,V]":
    #     return queries.Sleep(self, seconds)

    def zip(self):
        raise NotImplementedError()

    def cartesian(self, *iterables) -> "Query[Tuple]":
        return queries.Cartesian(self, *iterables)


if not TYPE_CHECKING:
    import types

    class Queries:
        pass

    # from .base import queries

    from ._itertools import queryables as queries

    classess = Queries()

    for cls in queries.exports:
        baseclasses = (cls, Query[T])
        created = types.new_class(cls.__name__, baseclasses)

        setattr(classess, cls.__name__, created)

    queries = classess


class QueryBase(Query[T], core.Query[T]):
    pass


class QueryAsync(Query[T], core.QueryAsync[T]):
    pass


class QueryNormal(Query[T], core.QueryNormal[T]):
    pass


class QueryDict(PairQuery[K, V], core.QueryDict[K, V]):
    def filter_keys(self, *keys) -> "PairQuery[K, V]":
        return queries.FilterKeys(self, *keys)

    # @no_type_check
    def must_keys(self, *keys) -> "PairQuery[K, V]":
        return queries.MustKeys(self, *keys, typ="map")

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
        return queries.FilterKeys(self, *keys)

    def must_keys(self, *keys) -> "Query[T]":
        return queries.MustKeys(self, *keys, typ="seq")

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
        return queries.FilterKeys(self, *keys)

    def must_keys(self, *keys) -> "Query[T]":
        return queries.MustKeys(self, *keys, typ="seq")

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
        return queries.FilterKeys(self, *keys)

    def must_keys(self, *keys) -> "Query[T]":
        return queries.MustKeys(self, *keys, typ="set")

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
