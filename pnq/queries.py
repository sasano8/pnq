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
    Literal,
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

from . import actions
from .core import LazyIterate as _LazyIterate
from .core import LazyReference as _LazyReference
from .core import piter, undefined
from .exceptions import NoElementError, NotFoundError, NotOneElementError
from .op import TH_ASSIGN_OP
from .requests import Response

T = TypeVar("T")
K = TypeVar("K")
V = TypeVar("V")
K2 = TypeVar("K2")
V2 = TypeVar("V2")
R = TypeVar("R")


__all__ = [
    "Query",
    "PairQuery",
    "IndexQuery",
    "ListEx",
    "DictEx",
    "SetEx",
    "query",
    "undefined",
]


def lazy_iterate(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        return LazyIterate(func, *args, **kwargs)

    return wrapper


def lazy_reference(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        return LazyReference(func, *args, **kwargs)

    return wrapper


if not TYPE_CHECKING:
    # TODO: type hintを文字で囲むのが面倒なため仮定義 文字化して除去する
    class Query(Generic[T], Iterable[T]):
        ...

    class PairQuery(Generic[K, V], Iterable[Tuple[K, V]]):
        ...

    class IndexQuery(Generic[K, V]):
        pass

    class ListEx(Generic[T], Sequence[T]):
        pass

    class DictEx(Generic[K, V], Mapping[K, V]):
        pass

    class SetEx(Generic[T], Iterable[T]):
        pass


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

    @lazy_iterate
    def enumerate(self, start: int = 0, step: int = 1) -> "PairQuery[int, T]":
        yield from (x * step for x in enumerate(self, start))

    @overload
    def map(self, selector: Callable[[T], Tuple[K2, V2]]) -> "PairQuery[K2, V2]":
        pass

    @overload
    def map(self, selector: Callable[[T], R]) -> "Query[R]":
        pass

    def map(self, selector):
        if selector is None:
            raise TypeError("selector cannot be None")
        return LazyIterate(actions.map, self, selector)

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
        if attr:
            selector = attrgetter(*fields)
        else:
            selector = itemgetter(*fields)
        return LazyIterate(actions.map, self, selector)

    def select_as_tuple(self, *fields, attr: bool = False) -> "Query[Tuple]":
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
        return LazyIterate(actions.map, self, selector)

    def select_as_dict(
        self, *fields, attr: bool = False, default=NoReturn
    ) -> "Query[Dict]":
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
        return LazyIterate(actions.map, self, selector)

    @lazy_iterate
    def unpack_pos(self, selector: Callable[..., R]) -> "Query[R]":
        return LazyIterate(actions.unpack_pos, self, selector)

    @lazy_iterate
    def unpack_kw(self, selector: Callable[..., R]) -> "Query[R]":
        return LazyIterate(actions.unpack_kw, self, selector)

    def group_by(
        self, selector: Callable[[T], Tuple[K2, V2]] = lambda x: x
    ) -> "PairQuery[K2, List[V2]]":
        return LazyIterate(actions.group_by, self, selector)

    def pivot_unstack(self, default=None) -> "PairQuery[Any, List]":
        return LazyIterate(actions.pivot_unstack, self, default)

    def pivot_stack(self) -> "Query[Dict]":
        return LazyIterate(actions.pivot_stack, self)

    def join(self, right, on: Callable[[Tuple[list, list]], Callable], select):
        [].join(
            [],
            lambda left, right: left.name == right.name,
            lambda left, right: (left.name, right.age),
        )

        table(User).join(Item, on=User.id == Item.id).select(User.id, Item.id)

        pass

    def request(self, func, retry: int = None) -> "Query[Response]":
        return LazyIterate(actions.request, self, func, retry)

    def request_async(self, func, retry: int = None) -> "Query[Response]":
        return LazyIterate(actions.request_async, self, func, retry)

    @lazy_iterate
    def distinct(self, selector: Callable[[T], Any], msg: str = ...) -> "Query[T]":
        duplicate = set()
        for elm in self:
            value = selector(elm)
            if not value in duplicate:
                duplicate.add(value)
                yield elm

    @lazy_iterate
    def zip(self):
        raise NotImplementedError()

    @lazy_iterate
    def filter(self, predicate: Callable[[T], bool]) -> "Query[T]":
        yield from filter(predicate, self)  # type: ignore

    @lazy_iterate
    def filter_type(self, *types: Type[R]) -> Query[T]:
        def normalize(type):
            if type is None:
                return None.__class__
            else:
                return type

        types = tuple((normalize(x) for x in types))
        return filter(lambda x: isinstance(x, *types), self)

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
        return LazyIterate(actions.filter_unique, self, selector)

    def must(self, predicate: Callable[[T], bool], msg: str = "") -> "Query[T]":
        """要素の検証に失敗した時例外を発生させる。"""
        return LazyIterate(actions.must, self, predicate, msg)

    def must_type(self, type, *types: Type) -> "Query[T]":
        """要素の検証に失敗した時例外を発生させる。"""
        return LazyIterate(actions.must_type, self, (type, *types))

    def must_unique(self, selector: Callable[[T], R] = None):
        return LazyIterate(actions.must_unique, self, selector)

    @lazy_iterate
    def skip(self, count: int) -> "Query[T]":
        current = 0

        try:
            while current < count:
                next(self)
                current += 1
        except StopIteration:
            return

        for elm in self:
            yield elm

    @lazy_iterate
    def take(self, count: int) -> "Query[T]":
        current = 0

        try:
            while current < count:
                yield next(self)
                current += 1
        except StopIteration:
            return

    def take_range(self, start: int = 0, stop: int = None) -> "Query[T]":
        return LazyIterate(actions.take_range, self, start=start, stop=stop)

    def take_page(self, page: int, size: int) -> "Query[T]":
        start, stop = actions.take_page_calc(page, size)
        return LazyIterate(actions.take_range, self, start=start, stop=stop)

    @lazy_reference
    def reverse(self) -> "Query[T]":
        yield actions.reverse(self)

    def order_by(self, *fields, desc: bool = False, attr: bool = False) -> "Query[T]":
        if not len(fields):
            selector = None
        else:
            if attr:
                selector = attrgetter(*fields)
            else:
                selector = itemgetter(*fields)

        return LazyIterate(actions.order_by, self, selector=selector, desc=desc)

    def order_by_map(self, selector=None, *, desc: bool = False) -> "Query[T]":
        return LazyIterate(actions.order_by, self, selector=selector, desc=desc)

    # @lazy_iterate
    # def order(self, selector, desc: bool = False) -> "Query[T]":
    #     yield from sorted(self, key=selector, reverse=desc)

    # def order_by_items(
    #     self, *items: Any, desc: bool = False
    # ) -> Query[T]:
    #     selector = itemgetter(*items)
    #     return self.order(selector, desc)

    # def order_by_attrs(
    #     self: Iterable[T], *attrs: str, desc: bool = False
    # ) -> Query[T]:
    #     selector = attrgetter(*attrs)
    #     return self.order(selector, desc)

    @lazy_iterate
    def sleep(self, seconds: float):
        from time import sleep

        for elm in self:
            yield elm
            sleep(seconds)

    @lazy_iterate
    async def sleep_async(self, seconds: float):
        from asyncio import sleep

        for elm in self:
            yield elm
            await sleep(seconds)

    class SyncAsync:
        def __init__(self, sync_func, async_func, *args, **kwargs) -> None:
            self.sync_func = sync_func
            self.async_func = async_func
            self.args = args
            self.kwargs = kwargs

        def __iter__(self):
            return self.async_func(*self.args, **self.kwargs)

        def __aiter__(self):
            return self.async_func(*self.args, **self.kwargs)

    def sleep2(self, seconds: float):
        from asyncio import sleep as asleep
        from time import sleep

        def sleep_sync(self, seconds):
            for elm in self:
                yield elm
                sleep(seconds)

        async def sleep_async(self, seconds):
            for elm in self:
                yield elm
                await sleep(seconds)

        return sleep_sync, sleep_async, seconds

    def debug(self, breakpoint=lambda x: x, printer=print) -> "Query[T]":
        return LazyIterate(actions.debug, self, breakpoint=breakpoint, printer=printer)

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

    @lazy_iterate
    def enumerate(self, start: int = 0, step: int = 1) -> "PairQuery[int, Tuple[K,V]]":
        yield from (x * step for x in enumerate(self, start))

    @overload
    def map(
        self, selector: Callable[[Tuple[K, V]], Tuple[K2, V2]]
    ) -> "PairQuery[K2, V2]":
        pass

    @overload
    def map(self, selector: Callable[[Tuple[K, V]], R]) -> "Query[R]":
        pass

    def map(self, selector):
        if selector is None:
            raise TypeError("selector cannot be None")
        return LazyIterate(actions.map, self, selector)

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
        if attr:
            selector = attrgetter(*fields)
        else:
            selector = itemgetter(*fields)
        return LazyIterate(actions.map, self, selector)

    def select_as_tuple(self, *fields, attr: bool = False) -> "Query[Tuple]":
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
        return LazyIterate(actions.map, self, selector)

    def select_as_dict(
        self, *fields, attr: bool = False, default=NoReturn
    ) -> "Query[Dict]":
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
        return LazyIterate(actions.map, self, selector)

    @lazy_iterate
    def unpack_pos(self, selector: Callable[..., R]) -> "Query[R]":
        return LazyIterate(actions.unpack_pos, self, selector)

    @lazy_iterate
    def unpack_kw(self, selector: Callable[..., R]) -> "Query[R]":
        return LazyIterate(actions.unpack_kw, self, selector)

    def group_by(
        self, selector: Callable[[Tuple[K, V]], Tuple[K2, V2]] = lambda x: x
    ) -> "PairQuery[K2, List[V2]]":
        return LazyIterate(actions.group_by, self, selector)

    def pivot_unstack(self, default=None) -> "PairQuery[Any, List]":
        return LazyIterate(actions.pivot_unstack, self, default)

    def pivot_stack(self) -> "Query[Dict]":
        return LazyIterate(actions.pivot_stack, self)

    def join(self, right, on: Callable[[Tuple[list, list]], Callable], select):
        [].join(
            [],
            lambda left, right: left.name == right.name,
            lambda left, right: (left.name, right.age),
        )

        table(User).join(Item, on=User.id == Item.id).select(User.id, Item.id)

        pass

    def request(self, func, retry: int = None) -> "Query[Response]":
        return LazyIterate(actions.request, self, func, retry)

    def request_async(self, func, retry: int = None) -> "Query[Response]":
        return LazyIterate(actions.request_async, self, func, retry)

    @lazy_iterate
    def distinct(
        self, selector: Callable[[Tuple[K, V]], Any], msg: str = ...
    ) -> "PairQuery[K,V]":
        duplicate = set()
        for elm in self:
            value = selector(elm)
            if not value in duplicate:
                duplicate.add(value)
                yield elm

    @lazy_iterate
    def zip(self):
        raise NotImplementedError()

    @lazy_iterate
    def filter(self, predicate: Callable[[Tuple[K, V]], bool]) -> "PairQuery[K,V]":
        yield from filter(predicate, self)  # type: ignore

    @lazy_iterate
    def filter_type(self, *types: Type[R]) -> PairQuery[K, V]:
        def normalize(type):
            if type is None:
                return None.__class__
            else:
                return type

        types = tuple((normalize(x) for x in types))
        return filter(lambda x: isinstance(x, *types), self)

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
        return LazyIterate(actions.filter_unique, self, selector)

    def must(
        self, predicate: Callable[[Tuple[K, V]], bool], msg: str = ""
    ) -> "PairQuery[K,V]":
        """要素の検証に失敗した時例外を発生させる。"""
        return LazyIterate(actions.must, self, predicate, msg)

    def must_type(self, type, *types: Type) -> "PairQuery[K,V]":
        """要素の検証に失敗した時例外を発生させる。"""
        return LazyIterate(actions.must_type, self, (type, *types))

    def must_unique(self, selector: Callable[[T], R] = None):
        return LazyIterate(actions.must_unique, self, selector)

    @lazy_iterate
    def skip(self, count: int) -> "PairQuery[K,V]":
        current = 0

        try:
            while current < count:
                next(self)
                current += 1
        except StopIteration:
            return

        for elm in self:
            yield elm

    @lazy_iterate
    def take(self, count: int) -> "PairQuery[K,V]":
        current = 0

        try:
            while current < count:
                yield next(self)
                current += 1
        except StopIteration:
            return

    def take_range(self, start: int = 0, stop: int = None) -> "PairQuery[K,V]":
        return LazyIterate(actions.take_range, self, start=start, stop=stop)

    def take_page(self, page: int, size: int) -> "PairQuery[K,V]":
        start, stop = actions.take_page_calc(page, size)
        return LazyIterate(actions.take_range, self, start=start, stop=stop)

    @lazy_reference
    def reverse(self) -> "PairQuery[K,V]":
        yield actions.reverse(self)

    def order_by(
        self, *fields, desc: bool = False, attr: bool = False
    ) -> "PairQuery[K,V]":
        if not len(fields):
            selector = None
        else:
            if attr:
                selector = attrgetter(*fields)
            else:
                selector = itemgetter(*fields)

        return LazyIterate(actions.order_by, self, selector=selector, desc=desc)

    def order_by_map(self, selector=None, *, desc: bool = False) -> "PairQuery[K,V]":
        return LazyIterate(actions.order_by, self, selector=selector, desc=desc)

    # @lazy_iterate
    # def order(self, selector, desc: bool = False) -> "PairQuery[K,V]":
    #     yield from sorted(self, key=selector, reverse=desc)

    # def order_by_items(
    #     self, *items: Any, desc: bool = False
    # ) -> PairQuery[K,V]:
    #     selector = itemgetter(*items)
    #     return self.order(selector, desc)

    # def order_by_attrs(
    #     self: Iterable[T], *attrs: str, desc: bool = False
    # ) -> PairQuery[K,V]:
    #     selector = attrgetter(*attrs)
    #     return self.order(selector, desc)

    @lazy_iterate
    def sleep(self, seconds: float):
        from time import sleep

        for elm in self:
            yield elm
            sleep(seconds)

    @lazy_iterate
    async def sleep_async(self, seconds: float):
        from asyncio import sleep

        for elm in self:
            yield elm
            await sleep(seconds)

    class SyncAsync:
        def __init__(self, sync_func, async_func, *args, **kwargs) -> None:
            self.sync_func = sync_func
            self.async_func = async_func
            self.args = args
            self.kwargs = kwargs

        def __iter__(self):
            return self.async_func(*self.args, **self.kwargs)

        def __aiter__(self):
            return self.async_func(*self.args, **self.kwargs)

    def sleep2(self, seconds: float):
        from asyncio import sleep as asleep
        from time import sleep

        def sleep_sync(self, seconds):
            for elm in self:
                yield elm
                sleep(seconds)

        async def sleep_async(self, seconds):
            for elm in self:
                yield elm
                await sleep(seconds)

        return sleep_sync, sleep_async, seconds

    def debug(self, breakpoint=lambda x: x, printer=print) -> "PairQuery[K,V]":
        return LazyIterate(actions.debug, self, breakpoint=breakpoint, printer=printer)

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


class Lazy:
    def len(self) -> int:
        return len(list(self))

    def exists(self) -> bool:
        return len(list(self)) > 0


class LazyIterate(Lazy, Query[T], _LazyIterate):
    pass


class LazyReference(Lazy, IndexQuery[int, T], Query[T], _LazyReference):
    pass


class Instance:
    def len(self) -> int:
        return len(self)

    def exists(self) -> bool:
        return len(self) > 0


class ListEx(Instance, IndexQuery[int, T], Query[T], List[T]):
    def __piter__(self):
        return self.__iter__()

    @lazy_reference
    def reverse(self) -> "Query[T]":
        yield from reversed(self)

    @no_type_check
    def get_many(self, *keys):
        return LazyReference(actions.get_many_for_sequence, self, *keys)

    @no_type_check
    def must_get_many(self, *keys):
        return LazyReference(actions.must_get_many, self, *keys, typ="seq")


class TupleEx(Instance, IndexQuery[int, T], Query[T], Tuple[T]):
    def __piter__(self):
        return self.__iter__()

    @lazy_reference
    def reverse(self) -> "Query[T]":
        yield from reversed(self)

    @no_type_check
    def get_many(self, *keys):
        return LazyReference(actions.get_many_for_sequence, self, *keys)

    @no_type_check
    def must_get_many(self, *keys):
        return LazyReference(actions.must_get_many, self, *keys, typ="seq")


class DictEx(Instance, IndexQuery[K, V], PairQuery[K, V], Dict[K, V]):
    def __piter__(self):
        return self.items().__iter__()

    @lazy_reference
    def keys(self):
        yield from super().keys()

    @lazy_reference
    def values(self):
        yield from super().values()

    @lazy_reference
    def items(self):
        yield from super().items()

    @lazy_reference
    def reverse(self) -> "PairQuery[K, V]":
        for key in reversed(self):
            yield key, self[key]

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

    @lazy_reference
    def reverse(self) -> "Query[T]":
        yield from reversed(self)

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

    @lazy_reference
    def reverse(self) -> "Query[T]":
        yield from reversed(self)

    @no_type_check
    def get_many(self, *keys):
        return LazyReference(actions.get_many_for_set, self, *keys)

    @no_type_check
    def must_get_many(self, *keys):
        return LazyReference(actions.must_get_many, self, *keys, typ="set")


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
    if hasattr(source, "__piter__"):
        return source
    elif isinstance(source, dict):
        return DictEx(source)
    elif isinstance(source, list):
        return ListEx(source)
    elif isinstance(source, tuple):
        return TupleEx(source)
    elif isinstance(source, set):
        return SetEx(source)
    elif isinstance(source, frozenset):
        return FrozenSetEx(source)
    elif hasattr(source, "__iter__"):
        return LazyIterate(iter, source)
    else:
        raise Exception()
