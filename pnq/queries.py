# type: ignore

from functools import wraps
from operator import attrgetter, itemgetter
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    FrozenSet,
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

from . import actions
from .base.exceptions import NoElementError, NotFoundError, NotOneElementError
from .base.op import TH_ASSIGN_OP
from .base.requests import Response

if TYPE_CHECKING:
    from .base import queries

T = TypeVar("T")
K = TypeVar("K")
V = TypeVar("V")
K2 = TypeVar("K2")
V2 = TypeVar("V2")
R = TypeVar("R")


__all__ = ["Query", "PairQuery", "IndexQuery", "ListEx", "DictEx", "SetEx", "query"]


class Query(Generic[T]):
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
    def to(self, func: Type[Iterable[T]]) -> Iterable[T]:
        ...

    @overload
    def to(self, func: Callable[[Iterable[T]], R]) -> R:
        ...

    def to(self, func: Callable[[Iterable[T]], R]) -> R:
        return actions.to(self, func)

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
        return queries.Map(self, selector)

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

    def unpack_pos(self, selector: Callable[..., R]) -> "Query[R]":
        return queries.UnpackPos(self, selector=selector)

    def unpack_kw(self, selector: Callable[..., R]) -> "Query[R]":
        return queries.UnpackKw(self, selector=selector)

    def group_by(
        self, selector: Callable[[T], Tuple[K2, V2]] = lambda x: x
    ) -> "PairQuery[K2, List[V2]]":
        return queries.GroupBy(self, selector=selector)

    def pivot_unstack(self, default=None) -> "PairQuery[Any, List]":
        return queries.PivotUnstack(self, default=default)

    def pivot_stack(self) -> "Query[Dict]":
        return queries.PivotStack(self)

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

    def zip(self):
        raise NotImplementedError()

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
        """要素の検証に失敗した時例外を発生させる。"""
        return queries.Must(self, predicate, msg)

    def must_type(self, type, *types: Type) -> "Query[T]":
        """要素の検証に失敗した時例外を発生させる。"""
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

    def sleep_async(self, seconds: float) -> "Query[T]":
        return queries.Sleep(self, seconds)

    def debug(self, breakpoint=lambda x: x, printer=print) -> "Query[T]":
        return queries.Debug(self, breakpoint=breakpoint, printer=printer)

    def debug_path(
        self, selector_sync=lambda x: -10, selector_async=lambda x: 10
    ) -> "Query[T]":
        return queries.DebugPath(self, selector_sync, selector_async)

    # if index query


class PairQuery(Generic[K, V]):
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
    def to(self, func: Type[Mapping[K, V]]) -> Mapping[K, V]:
        ...

    @overload
    def to(self, func: Callable[[Iterable[Tuple[K, V]]], R]) -> R:
        ...

    @overload
    def to(self, func: Type[Iterable[T]]) -> Iterable[T]:
        ...

    @overload
    def to(self, func: Callable[[Iterable[T]], R]) -> R:
        ...

    def to(self, func: Callable[[Iterable[T]], R]) -> R:
        return actions.to(self, func)

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
        return queries.Map(self, selector)

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

    def unpack_pos(self, selector: Callable[..., R]) -> "Query[R]":
        return queries.UnpackPos(self, selector=selector)

    def unpack_kw(self, selector: Callable[..., R]) -> "Query[R]":
        return queries.UnpackKw(self, selector=selector)

    def group_by(
        self, selector: Callable[[Tuple[K, V]], Tuple[K2, V2]] = lambda x: x
    ) -> "PairQuery[K2, List[V2]]":
        return queries.GroupBy(self, selector=selector)

    def pivot_unstack(self, default=None) -> "PairQuery[Any, List]":
        return queries.PivotUnstack(self, default=default)

    def pivot_stack(self) -> "Query[Dict]":
        return queries.PivotStack(self)

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

    def zip(self):
        raise NotImplementedError()

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
        """要素の検証に失敗した時例外を発生させる。"""
        return queries.Must(self, predicate, msg)

    def must_type(self, type, *types: Type) -> "PairQuery[K,V]":
        """要素の検証に失敗した時例外を発生させる。"""
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

    def sleep_async(self, seconds: float) -> "PairQuery[K,V]":
        return queries.Sleep(self, seconds)

    def debug(self, breakpoint=lambda x: x, printer=print) -> "PairQuery[K,V]":
        return queries.Debug(self, breakpoint=breakpoint, printer=printer)

    def debug_path(
        self, selector_sync=lambda x: -10, selector_async=lambda x: 10
    ) -> "PairQuery[K,V]":
        return queries.DebugPath(self, selector_sync, selector_async)

    # if index query


class IndexQuery(Generic[K, V]):
    def get_many(self, *keys) -> "IndexQuery[K,V]":
        raise NotImplementedError()

    def must_get_many(self, *keys) -> "IndexQuery[K,V]":
        raise NotImplementedError()

    @overload
    def get(self, key: K) -> V:
        ...

    @overload
    def get(self, key: K, default: R = NoReturn) -> Union[V, R]:
        ...

    def get(self, key: K, default=NoReturn) -> Any:
        return actions.get(self, key, default)

    def get_or(self, key: K, default: R) -> Union[V, R]:
        return actions.get_or(self, key, default)

    def get_or_raise(self, key: K, exc: Union[str, Exception]) -> V:
        return actions.get_or_raise(self, key, exc)


# 継承時は右側に基底クラスを指定し、左へ上書きしていくイメージ

# class Lazy:
#     def len(self) -> int:
#         return len(list(self))

#     def exists(self) -> bool:
#         return len(list(self)) > 0


# class LazyIterate(Lazy, Query[T], _LazyIterate):
#     pass


# class LazyReference(Lazy, IndexQuery[int, T], Query[T], _LazyReference):
#     pass


class Instance:
    def len(self) -> int:
        return len(self)

    def exists(self) -> bool:
        return len(self) > 0


class ListEx(Instance, IndexQuery[int, T], Query[T], List[T]):
    def __piter__(self):
        return self.__iter__()

    @no_type_check
    def get_many(self, *keys):
        return LazyReference(actions.get_many_for_sequence, self, *keys)

    @no_type_check
    def must_get_many(self, *keys):
        return LazyReference(actions.must_get_many, self, *keys, typ="seq")


class TupleEx(Instance, IndexQuery[int, T], Query[T], Tuple[T]):
    def __piter__(self):
        return self.__iter__()

    @no_type_check
    def get_many(self, *keys):
        return LazyReference(actions.get_many_for_sequence, self, *keys)

    @no_type_check
    def must_get_many(self, *keys):
        return LazyReference(actions.must_get_many, self, *keys, typ="seq")


class DictEx(Instance, IndexQuery[K, V], PairQuery[K, V], Dict[K, V]):
    def __piter__(self):
        return self.items().__iter__()

    # @lazy_reference
    # def keys(self):
    #     yield from super().keys()

    # @lazy_reference
    # def values(self):
    #     yield from super().values()

    # @lazy_reference
    # def items(self):
    #     yield from super().items()

    # @lazy_reference
    # def reverse(self) -> "PairQuery[K, V]":
    #     for key in reversed(self):
    #         yield key, self[key]

    @no_type_check
    def get_many(self, *keys):
        return LazyReference(actions.get_many_for_mapping, self, *keys)

    @no_type_check
    def must_get_many(self, *keys):
        return LazyReference(actions.must_get_many, self, *keys, typ="map")


class SetEx(Instance, IndexQuery[T, T], Query[T], Set[T]):
    def __piter__(self):
        return self.__iter__()

    def __getitem__(self, key: T):
        if key in self:
            return key
        else:
            raise NotFoundError(key)

    def order_by_reverse(self) -> "Query[T]":
        raise NotImplementedError("Set has no order.")

    @no_type_check
    def get_many(self, *keys):
        return LazyReference(actions.get_many_for_set, self, *keys)

    @no_type_check
    def must_get_many(self, *keys):
        return LazyReference(actions.must_get_many, self, *keys, typ="set")


class FrozenSetEx(Instance, IndexQuery[T, T], Query[T], FrozenSet[T]):
    def __piter__(self):
        return self.__iter__()

    def __getitem__(self, key: T):
        if key in self:
            return key
        else:
            raise NotFoundError(key)

    def order_by_reverse(self) -> "Query[T]":
        raise NotImplementedError("Set has no order.")

    @no_type_check
    def get_many(self, *keys):
        return LazyReference(actions.get_many_for_set, self, *keys)

    @no_type_check
    def must_get_many(self, *keys):
        return LazyReference(actions.must_get_many, self, *keys, typ="set")


if TYPE_CHECKING:
    from .base import queries

else:
    import types

    class Queries:
        pass

    from .base import queries

    classess = Queries()

    for cls in queries.exports:
        baseclasses = (cls, Query[T])
        created = types.new_class(cls.__name__, baseclasses)

        setattr(classess, cls.__name__, created)

    queries = classess

from .base import builder


class QueryDict(queries.QueryDict):
    def get_many(self, *keys):
        return queries.FilterKeys(self, *keys)

    def get(self, key):
        try:
            return self.source[key]
        except KeyError:
            raise NotFoundError(key)

    def get_or(self, key, default=None):
        try:
            return self.get(key)
        except NotFoundError:
            return default

    def get_or_raise(self, key, exc: Union[str, Exception]):
        undefined = object()
        result = self.get_or(key, undefined)
        if result is undefined:
            if isinstance(exc, str):
                raise Exception(exc)
            else:
                raise exc
        else:
            return result

    # @no_type_check
    def must_get_many(self, *keys):
        return queries.MustKeys(self, *keys, typ="map")


class QuerySeq(queries.QuerySeq):
    def get_many(self, *keys):
        return queries.FilterKeys(self, *keys)

    def get(self, key):
        try:
            return self.source[key]
        except IndexError:
            raise NotFoundError(key)

    def get_or(self, key, default=None):
        try:
            return self.get(key)
        except NotFoundError:
            return default

    def get_or_raise(self, key, exc: Union[str, Exception]):
        undefined = object()
        result = self.get_or(key, undefined)
        if result is undefined:
            if isinstance(exc, str):
                raise Exception(exc)
            else:
                raise exc
        else:
            return result

    # @no_type_check
    def must_get_many(self, *keys):
        return queries.MustKeys(self, *keys, typ="seq")


class QuerySet(queries.QuerySet):
    def order_by_reverse(self) -> "Query[T]":
        raise NotImplementedError("Set has no order.")

    def get_many(self, *keys):
        return queries.FilterKeys(self, *keys)

    def get(self, key):
        if key in self.source:
            return key
        else:
            raise NotFoundError(key)

    def get_or(self, key, default=None):
        try:
            return self.get(key)
        except NotFoundError:
            return default

    def get_or_raise(self, key, exc: Union[str, Exception]):
        undefined = object()
        result = self.get_or(key, undefined)
        if result is undefined:
            if isinstance(exc, str):
                raise Exception(exc)
            else:
                raise exc
        else:
            return result

    # @no_type_check
    def must_get_many(self, *keys):
        return queries.MustKeys(self, *keys, typ="set")


class QueryBuilder(builder.Builder):
    QUERY_BOTH = queries.Query
    QUERY_ASYNC = queries.QueryAsync
    QUERY_NORMAL = queries.QueryNormal
    QUERY_SEQ = QuerySeq
    QUERY_DICT = QueryDict
    QUERY_SET = QuerySet


@overload
def query(source: Mapping[K, V]) -> DictEx[K, V]:
    ...


@overload
def query(source: Iterable[T]) -> ListEx[T]:
    ...


@overload
def query(source) -> "Query[Any]":
    ...


def query(source):
    return QueryBuilder.query(source)


run = QueryBuilder.run
