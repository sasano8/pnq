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


class Query(Generic[T]):
    if TYPE_CHECKING:

        def __iter__(self) -> Iterator[T]:
            ...

        def __aiter__(self) -> AsyncIterator[T]:
            ...

    def as_iter(self) -> "Finalizer[T]":
        # return finalizers.SyncFinalizer(self)
        return Finalizer(self)

    def as_aiter(self) -> "AsyncFinalizer[T]":
        return AsyncFinalizer(self)

    to_file = Finalizer.to_file
    to_csv = Finalizer.to_csv
    to_json = Finalizer.to_json
    to_jsonl = Finalizer.to_jsonl

    @property
    def _(self) -> "AsyncFinalizer[T]":
        return AsyncFinalizer(self)

    async def __aresult__(self):
        return await PnqList.from_aiter(self)

    def result(self, timeout=None) -> "PnqList[T]":
        return PnqList(self)

    save = result

    def __await__(self) -> Generator[Any, Any, "PnqList[T]"]:
        return self.__aresult__().__await__()

    def len(self) -> int:
        return Finalizer.len(self)

    def exists(self, predicate=None) -> bool:
        return Finalizer.exists(self, predicate)

    def all(self, selector: Callable[[T], Any] = lambda x: x) -> bool:
        return Finalizer.all(self, selector)

    def any(self, selector: Callable[[T], Any] = lambda x: x) -> bool:
        return Finalizer.any(self, selector)

    def contains(self, value, selector: Callable[[T], Any] = lambda x: x) -> bool:
        return Finalizer.contains(self, value, selector)

    @overload
    def min(self, *, default=NoReturn) -> Union[T, NoReturn]:
        ...

    @overload
    def min(self, key_selector: Callable[[T], R] = lambda x: x, default=NoReturn) -> R:
        ...

    def min(self, key_selector: Callable[[T], R] = lambda x: x, default=NoReturn) -> R:
        return Finalizer.min(self, key_selector, default)

    @overload
    def max(self, *, default=NoReturn) -> Union[T, NoReturn]:
        ...

    @overload
    def max(self, key_selector: Callable[[T], R] = lambda x: x, default=NoReturn) -> R:
        ...

    def max(self, key_selector: Callable[[T], R] = lambda x: x, default=NoReturn) -> R:
        return Finalizer.max(self, key_selector, default)

    @overload
    def sum(self) -> T:
        ...

    @overload
    def sum(self, selector: Callable[[T], R] = lambda x: x) -> R:
        ...

    def sum(self, selector: Callable[[T], R] = lambda x: x) -> R:
        return Finalizer.sum(self, selector)

    @overload
    def average(self) -> T:
        ...

    @overload
    def average(self, selector: Callable[[T], R] = lambda x: x) -> R:
        ...

    def average(self, selector: Callable[[T], R] = lambda x: x) -> R:
        return Finalizer.average(self, selector)

    def reduce(
        self,
        seed: T,
        op: Union[TH_ASSIGN_OP, Callable[[Any, Any], Any]] = "+=",
        selector=lambda x: x,
    ) -> T:
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
            return_sync = sync_result == expect

        if async_result is not None:
            return_async = async_result == expect

        return return_sync, return_async

    def gather(self):
        return queryables.Gather(self)

    def request(
        self,
        func,
        executor=None,
        *,
        unpack="",
        chunksize=1,
        retry: int = None,
        timeout: float = None,
    ) -> "Query[Response]":
        return queryables.Request(
            self,
            func,
            executor,
            unpack=unpack,
            chunksize=chunksize,
            retry=retry,
            timeout=timeout,
        )

    @overload
    def parallel(
        self,
        func: Callable[..., Awaitable[R]],
        executor=None,
        *,
        unpack="",
        chunksize=1,
    ) -> "Query[R]":
        ...

    @overload
    def parallel(
        self, func: Callable[..., R], executor=None, *, unpack="", chunksize=1
    ) -> "Query[R]":
        ...

    def parallel(self, func, executor=None, *, unpack="", chunksize=1) -> "Query[R]":
        return queryables.Parallel(
            self, func, executor, unpack=unpack, chunksize=chunksize
        )

    def dispatch(
        self,
        func,
        executor: "PExecutor" = None,
        *,
        unpack="",
        chunksize=1,
        on_complete=None,
    ):
        return Finalizer.dispatch(
            self,
            func,
            executor,
            unpack=unpack,
            chunksize=chunksize,
            on_complete=on_complete,
        )

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

    def each(self, func: Callable = lambda x: x, unpack=""):
        return Finalizer.each(self, func, unpack)

    def one(self) -> T:
        return Finalizer.one(self)

    def one_or(self, default: R) -> Union[T, R]:
        return Finalizer.one_or(self, default)

    def one_or_raise(self, exc: Union[str, Exception]) -> T:
        return Finalizer.one_or_raise(self, exc)

    def first(self) -> T:
        return Finalizer.first(self)

    def first_or(self, default: R) -> Union[T, R]:
        return Finalizer.first_or(self, default)

    def first_or_raise(self, exc: Union[str, Exception]) -> T:
        return Finalizer.first_or_raise(self, exc)

    def last(self) -> T:
        return Finalizer.last(self)

    def last_or(self, default: R) -> Union[T, R]:
        return Finalizer.last_or(self, default)

    def last_or_raise(self, exc: Union[str, Exception]) -> T:
        return Finalizer.last_or_raise(self, exc)

    @overload
    def cast(self, type: Type[Tuple[K2, V2]]) -> "PairQuery[K2, V2]":
        pass

    @overload
    def cast(self, type: Type[R]) -> "Query[R]":
        pass

    def cast(self, type: Type[R]) -> "Query[R]":
        return self

    def enumerate(self, start: int = 0, step: int = 1) -> "PairQuery[int, T]":
        return queryables.Enumerate(self, start, step)

    @overload
    def map(
        self, selector: Callable[[T], Tuple[K2, V2]], unpack=""
    ) -> "PairQuery[K2, V2]":
        pass

    @overload
    def map(self, selector: Callable[[T], R], unpack="") -> "Query[R]":
        pass

    @overload
    def map(self, selector: Callable[..., R], unpack="*") -> "Query[R]":
        pass

    @overload
    def map(self, selector: Callable[..., R], unpack="**") -> "Query[R]":
        pass

    @overload
    def map(self, selector: Callable[..., R], unpack="***") -> "Query[R]":
        pass

    def map(self, selector, unpack=""):
        return queryables.Map(self, selector, unpack)

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

    def select_as_dict(
        self, *fields, attr: bool = False, default=NoReturn
    ) -> "Query[Dict]":
        return queryables.SelectAsDict(self, *fields, attr=attr, default=default)

    def reflect(self, mapping, default=NoReturn, attr: bool = False):
        return queryables.Reflect(self, mapping, attr=attr)

    def flat(self, selector: Callable[..., Iterable[R]] = None) -> "Query[R]":
        return queryables.Flat(self, selector)

    def traverse(self, selector: Callable[[T], Iterable[T]]) -> "Query[T]":
        return queryables.Traverse(self, selector)

    def pivot_unstack(self, default=None) -> "PairQuery[Any, List]":
        return queryables.PivotUnstack(self, default=default)

    def pivot_stack(self) -> "Query[Dict]":
        return queryables.PivotStack(self)

    def group_by(
        self, selector: Callable[[T], Tuple[K2, V2]] = lambda x: x
    ) -> "PairQuery[K2, List[V2]]":
        return queryables.GroupBy(self, selector=selector)

    def chain(self, *iterables):
        return queryables.Chain(self, *iterables)

    def chunk(self, size: int) -> "Query[List[T]]":
        return queryables.Chunk(self, size=size)

    def tee(self, size: int):
        return queryables.Tee(self, size=size)

    def inner_join(self, right):
        return queryables.InnerJoin(self, right)

    def join(self, right, on: Callable[[Tuple[list, list]], Callable], select):
        return queryables.Join(self, right, on=on, select=select)

    def debug(self, breakpoint=lambda x: x, printer=print) -> "Query[T]":
        return queryables.Debug(self, breakpoint=breakpoint, printer=printer)

    def debug_path(
        self, selector_sync=lambda x: -10, selector_async=lambda x: 10
    ) -> "Query[T]":
        return queryables.DebugPath(self, selector_sync, selector_async)

    def filter(self, predicate: Callable[[T], bool]) -> "Query[T]":
        return queryables.Filter(self, predicate)

    def filter_type(self, *types: Type[R]) -> "Query[T]":
        return queryables.FilterType(self, *types)

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
        return queryables.FilterUnique(self, selector=selector)

    def distinct(self, selector: Callable[[T], Any]) -> "Query[T]":
        return queryables.FilterUnique(self, selector=selector)

    def must(self, predicate: Callable[[T], bool], msg: str = "") -> "Query[T]":
        return queryables.Must(self, predicate, msg)

    def must_type(self, type, *types: Type) -> "Query[T]":
        return queryables.MustType(self, type, *types)

    def must_unique(self, selector: Callable[[T], R] = None):
        return queryables.MustUnique(self, selector=selector)

    def take(self, count_or_range: Union[int, range]) -> "Query[T]":
        return queryables.Take(self, count_or_range)

    def take_while(self, predicate) -> "Query[T]":
        return queryables.TakeWhile(self, predicate)

    def skip(self, count_or_range: Union[int, range]) -> "Query[T]":
        return queryables.Skip(self, count_or_range)

    def take_page(self, page: int, size: int) -> "Query[T]":
        return queryables.TakePage(self, page=page, size=size)

    def order_by(self, *fields, desc: bool = False, attr: bool = False) -> "Query[T]":
        return queryables.OrderBy(self, *fields, desc=desc, attr=attr)

    def order_by_map(self, selector=None, *, desc: bool = False) -> "Query[T]":
        return queryables.OrderByMap(self, selector=selector, desc=desc)

    def order_by_reverse(self) -> "Query[T]":
        return queryables.OrderByReverse(self)

    def order_by_shuffle(self) -> "Query[T]":
        return queryables.OrderByShuffle(self)

    def sleep(self, seconds: float) -> "Query[T]":
        return queryables.Sleep(self, seconds)

    def zip(self):
        return queryables.Zip(self)

    def cartesian(self, *iterables) -> "Query[Tuple]":
        return queryables.Cartesian(self, *iterables)


class PairQuery(Generic[K, V], Query[Tuple[K, V]]):
    if TYPE_CHECKING:

        def __iter__(self) -> Iterator[Tuple[K, V]]:
            ...

        def __aiter__(self) -> AsyncIterator[Tuple[K, V]]:
            ...

    def as_iter(self) -> "Finalizer[Tuple[K,V]]":
        # return finalizers.SyncFinalizer(self)
        return Finalizer(self)

    def as_aiter(self) -> "AsyncFinalizer[Tuple[K,V]]":
        return AsyncFinalizer(self)

    to_file = Finalizer.to_file
    to_csv = Finalizer.to_csv
    to_json = Finalizer.to_json
    to_jsonl = Finalizer.to_jsonl

    @property
    def _(self) -> "AsyncFinalizer[Tuple[K,V]]":
        return AsyncFinalizer(self)

    async def __aresult__(self):
        return await PnqList.from_aiter(self)

    def result(self, timeout=None) -> "PnqListPair[K, V]":
        return PnqList(self)

    save = result

    def __await__(self) -> Generator[Any, Any, "PnqListPair[K, V]"]:
        return self.__aresult__().__await__()

    def len(self) -> int:
        return Finalizer.len(self)

    def exists(self, predicate=None) -> bool:
        return Finalizer.exists(self, predicate)

    def all(self, selector: Callable[[Tuple[K, V]], Any] = lambda x: x) -> bool:
        return Finalizer.all(self, selector)

    def any(self, selector: Callable[[Tuple[K, V]], Any] = lambda x: x) -> bool:
        return Finalizer.any(self, selector)

    def contains(
        self, value, selector: Callable[[Tuple[K, V]], Any] = lambda x: x
    ) -> bool:
        return Finalizer.contains(self, value, selector)

    @overload
    def min(self, *, default=NoReturn) -> Union[Tuple[K, V], NoReturn]:
        ...

    @overload
    def min(
        self, key_selector: Callable[[Tuple[K, V]], R] = lambda x: x, default=NoReturn
    ) -> R:
        ...

    def min(
        self, key_selector: Callable[[Tuple[K, V]], R] = lambda x: x, default=NoReturn
    ) -> R:
        return Finalizer.min(self, key_selector, default)

    @overload
    def max(self, *, default=NoReturn) -> Union[Tuple[K, V], NoReturn]:
        ...

    @overload
    def max(
        self, key_selector: Callable[[Tuple[K, V]], R] = lambda x: x, default=NoReturn
    ) -> R:
        ...

    def max(
        self, key_selector: Callable[[Tuple[K, V]], R] = lambda x: x, default=NoReturn
    ) -> R:
        return Finalizer.max(self, key_selector, default)

    @overload
    def sum(self) -> Tuple[K, V]:
        ...

    @overload
    def sum(self, selector: Callable[[Tuple[K, V]], R] = lambda x: x) -> R:
        ...

    def sum(self, selector: Callable[[Tuple[K, V]], R] = lambda x: x) -> R:
        return Finalizer.sum(self, selector)

    @overload
    def average(self) -> Tuple[K, V]:
        ...

    @overload
    def average(self, selector: Callable[[Tuple[K, V]], R] = lambda x: x) -> R:
        ...

    def average(self, selector: Callable[[Tuple[K, V]], R] = lambda x: x) -> R:
        return Finalizer.average(self, selector)

    def reduce(
        self,
        seed: T,
        op: Union[TH_ASSIGN_OP, Callable[[Any, Any], Any]] = "+=",
        selector=lambda x: x,
    ) -> T:
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
            return_sync = sync_result == expect

        if async_result is not None:
            return_async = async_result == expect

        return return_sync, return_async

    def gather(self):
        return queryables.Gather(self)

    def request(
        self,
        func,
        executor=None,
        *,
        unpack="",
        chunksize=1,
        retry: int = None,
        timeout: float = None,
    ) -> "Query[Response]":
        return queryables.Request(
            self,
            func,
            executor,
            unpack=unpack,
            chunksize=chunksize,
            retry=retry,
            timeout=timeout,
        )

    @overload
    def parallel(
        self,
        func: Callable[..., Awaitable[R]],
        executor=None,
        *,
        unpack="",
        chunksize=1,
    ) -> "Query[R]":
        ...

    @overload
    def parallel(
        self, func: Callable[..., R], executor=None, *, unpack="", chunksize=1
    ) -> "Query[R]":
        ...

    def parallel(self, func, executor=None, *, unpack="", chunksize=1) -> "Query[R]":
        return queryables.Parallel(
            self, func, executor, unpack=unpack, chunksize=chunksize
        )

    def dispatch(
        self,
        func,
        executor: "PExecutor" = None,
        *,
        unpack="",
        chunksize=1,
        on_complete=None,
    ):
        return Finalizer.dispatch(
            self,
            func,
            executor,
            unpack=unpack,
            chunksize=chunksize,
            on_complete=on_complete,
        )

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
        return Finalizer.to(self, func)

    def lazy(self, func, *args, **kwargs):
        return Finalizer.lazy(self, func, *args, **kwargs)

    def each(self, func: Callable = lambda x: x, unpack=""):
        return Finalizer.each(self, func, unpack)

    def one(self) -> Tuple[K, V]:
        return Finalizer.one(self)

    def one_or(self, default: R) -> Union[Tuple[K, V], R]:
        return Finalizer.one_or(self, default)

    def one_or_raise(self, exc: Union[str, Exception]) -> Tuple[K, V]:
        return Finalizer.one_or_raise(self, exc)

    def first(self) -> Tuple[K, V]:
        return Finalizer.first(self)

    def first_or(self, default: R) -> Union[Tuple[K, V], R]:
        return Finalizer.first_or(self, default)

    def first_or_raise(self, exc: Union[str, Exception]) -> Tuple[K, V]:
        return Finalizer.first_or_raise(self, exc)

    def last(self) -> Tuple[K, V]:
        return Finalizer.last(self)

    def last_or(self, default: R) -> Union[Tuple[K, V], R]:
        return Finalizer.last_or(self, default)

    def last_or_raise(self, exc: Union[str, Exception]) -> Tuple[K, V]:
        return Finalizer.last_or_raise(self, exc)

    @overload
    def cast(self, type: Type[Tuple[K2, V2]]) -> "PairQuery[K2, V2]":
        pass

    @overload
    def cast(self, type: Type[R]) -> "Query[R]":
        pass

    def cast(self, type: Type[R]) -> "Query[R]":
        return self

    def enumerate(self, start: int = 0, step: int = 1) -> "PairQuery[int, Tuple[K,V]]":
        return queryables.Enumerate(self, start, step)

    @overload
    def map(
        self, selector: Callable[[Tuple[K, V]], Tuple[K2, V2]], unpack=""
    ) -> "PairQuery[K2, V2]":
        pass

    @overload
    def map(self, selector: Callable[[Tuple[K, V]], R], unpack="") -> "Query[R]":
        pass

    @overload
    def map(self, selector: Callable[..., R], unpack="*") -> "Query[R]":
        pass

    @overload
    def map(self, selector: Callable[..., R], unpack="**") -> "Query[R]":
        pass

    @overload
    def map(self, selector: Callable[..., R], unpack="***") -> "Query[R]":
        pass

    def map(self, selector, unpack=""):
        return queryables.Map(self, selector, unpack)

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
        return queryables.Select(self, *fields, attr=attr)

    def select_as_tuple(self, *fields, attr: bool = False) -> "Query[Tuple]":
        return queryables.SelectAsTuple(self, *fields, attr=attr)

    def select_as_dict(
        self, *fields, attr: bool = False, default=NoReturn
    ) -> "Query[Dict]":
        return queryables.SelectAsDict(self, *fields, attr=attr, default=default)

    def reflect(self, mapping, default=NoReturn, attr: bool = False):
        return queryables.Reflect(self, mapping, attr=attr)

    def flat(self, selector: Callable[..., Iterable[R]] = None) -> "Query[R]":
        return queryables.Flat(self, selector)

    def traverse(
        self, selector: Callable[[Tuple[K, V]], Iterable[Tuple[K, V]]]
    ) -> "Query[Tuple[K,V]]":
        return queryables.Traverse(self, selector)

    def pivot_unstack(self, default=None) -> "PairQuery[Any, List]":
        return queryables.PivotUnstack(self, default=default)

    def pivot_stack(self) -> "Query[Dict]":
        return queryables.PivotStack(self)

    def group_by(
        self, selector: Callable[[Tuple[K, V]], Tuple[K2, V2]] = lambda x: x
    ) -> "PairQuery[K2, List[V2]]":
        return queryables.GroupBy(self, selector=selector)

    def chain(self, *iterables):
        return queryables.Chain(self, *iterables)

    def chunk(self, size: int) -> "Query[List[Tuple[K,V]]]":
        return queryables.Chunk(self, size=size)

    def tee(self, size: int):
        return queryables.Tee(self, size=size)

    def inner_join(self, right):
        return queryables.InnerJoin(self, right)

    def join(self, right, on: Callable[[Tuple[list, list]], Callable], select):
        return queryables.Join(self, right, on=on, select=select)

    def debug(self, breakpoint=lambda x: x, printer=print) -> "PairQuery[K,V]":
        return queryables.Debug(self, breakpoint=breakpoint, printer=printer)

    def debug_path(
        self, selector_sync=lambda x: -10, selector_async=lambda x: 10
    ) -> "PairQuery[K,V]":
        return queryables.DebugPath(self, selector_sync, selector_async)

    def filter(self, predicate: Callable[[Tuple[K, V]], bool]) -> "PairQuery[K,V]":
        return queryables.Filter(self, predicate)

    def filter_type(self, *types: Type[R]) -> "PairQuery[K,V]":
        return queryables.FilterType(self, *types)

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
        return queryables.FilterUnique(self, selector=selector)

    def distinct(self, selector: Callable[[Tuple[K, V]], Any]) -> "PairQuery[K,V]":
        return queryables.FilterUnique(self, selector=selector)

    def must(
        self, predicate: Callable[[Tuple[K, V]], bool], msg: str = ""
    ) -> "PairQuery[K,V]":
        return queryables.Must(self, predicate, msg)

    def must_type(self, type, *types: Type) -> "PairQuery[K,V]":
        return queryables.MustType(self, type, *types)

    def must_unique(self, selector: Callable[[T], R] = None):
        return queryables.MustUnique(self, selector=selector)

    def take(self, count_or_range: Union[int, range]) -> "PairQuery[K,V]":
        return queryables.Take(self, count_or_range)

    def take_while(self, predicate) -> "PairQuery[K,V]":
        return queryables.TakeWhile(self, predicate)

    def skip(self, count_or_range: Union[int, range]) -> "PairQuery[K,V]":
        return queryables.Skip(self, count_or_range)

    def take_page(self, page: int, size: int) -> "PairQuery[K,V]":
        return queryables.TakePage(self, page=page, size=size)

    def order_by(
        self, *fields, desc: bool = False, attr: bool = False
    ) -> "PairQuery[K,V]":
        return queryables.OrderBy(self, *fields, desc=desc, attr=attr)

    def order_by_map(self, selector=None, *, desc: bool = False) -> "PairQuery[K,V]":
        return queryables.OrderByMap(self, selector=selector, desc=desc)

    def order_by_reverse(self) -> "PairQuery[K,V]":
        return queryables.OrderByReverse(self)

    def order_by_shuffle(self) -> "PairQuery[K,V]":
        return queryables.OrderByShuffle(self)

    def sleep(self, seconds: float) -> "PairQuery[K,V]":
        return queryables.Sleep(self, seconds)

    def zip(self):
        return queryables.Zip(self)

    def cartesian(self, *iterables) -> "Query[Tuple]":
        return queryables.Cartesian(self, *iterables)


if not TYPE_CHECKING:
    import types

    class Queries:
        pass

    from ._itertools import queryables

    classess = Queries()

    for cls in queryables.exports:
        baseclasses = (Query[T], cls)
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
