from functools import wraps
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Generic,
    Iterable,
    Iterator,
    Mapping,
    Tuple,
    TypeVar,
    Union,
    overload,
)

from .exceptions import NoElementError, NotOneError

T = TypeVar("T")
K = TypeVar("K")
V = TypeVar("V")
R = TypeVar("R")
KV = TypeVar("KV", bound=Tuple[Any, Any], covariant=True)
F = TypeVar("F", bound=Callable)
undefined = object()


class Lazy(Generic[T]):
    def __init__(self, func: Callable[..., Iterator[T]], *args, **kwargs):
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def __call__(self) -> Iterator[T]:
        return self.func(*self.args, **self.kwargs)

    def __iter__(self) -> Iterator[T]:
        yield from self()

    @classmethod
    def decolate(cls, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return cls(func, *args, **kwargs)

        return wrapper


lazy = Lazy.decolate


class Query(Iterable[T]):
    def __init__(self, source: Iterable[T]):
        if isinstance(source, Mapping):
            raise TypeError()

        self.source = source

    def __iter__(self) -> Iterator[T]:
        return self.source.__iter__()

    def len(self):
        return len(list(self))

    @staticmethod
    def _first(source):
        it = iter(source)
        try:
            obj = next(it)
        except StopIteration:
            raise NoElementError()

        return it, obj

    @staticmethod
    def _last(source: Iterable):
        undefined = object()
        last = undefined
        for elm in source:
            last = elm

        if last is undefined:
            raise NoElementError()

        return last

    def first(self):
        it, obj = self._first(self)
        return obj

    def first_or_default(self, default: R = None) -> Union[T, R, None]:
        try:
            return self.first()
        except NoElementError:
            return default

    def one(self) -> T:
        it, obj = self._first(self)
        try:
            next(it)
        except StopIteration:
            return obj

        raise NotOneError()

    def one_or_default(self, default: R = None) -> Union[T, R, None]:
        try:
            return self.one()
        except NotOneError:
            return default
        except NoElementError:
            return default

    def last(self):
        return self._last(self)

    def last_or_default(self, default: R = None) -> Union[T, R, None]:
        try:
            return self.last()
        except NoElementError:
            return default

    @staticmethod
    @lazy
    def _order(source, selector, desc: bool = False):
        yield from sorted(source, key=selector, reverse=desc)

    def order(self: Iterable[T], selector, desc: bool = False) -> "Query[T]":
        return Query(Query._order(self, selector, desc))

    def order_by_attrs(self: Iterable[T], *attrs: str, desc: bool = False):
        if len(attrs) == 0:
            raise ValueError("required attrs at least one.")
        elif len(attrs) == 1:
            key = attrs[0]
            selector = lambda x: getattr(x, key)
        else:
            selector = lambda x: tuple((getattr(x, key) for key in attrs))

        return Query.order(self, selector, desc=desc)

    def order_by_index(self: Iterable[T], *indexes: Any, desc: bool = False):
        if len(indexes) == 0:
            raise ValueError("required indexes at least one.")
        elif len(indexes) == 1:
            key = indexes[0]
            selector = lambda x: x[key]
        else:
            selector = lambda x: tuple((x[key] for key in indexes))

        return Query.order(self, selector, desc=desc)

    def __reversed__(self: Iterable[T]):
        arr = list(self)
        yield from reversed(arr)

    @staticmethod
    @lazy
    def _reverse(source: Iterable[T]) -> Iterator[T]:
        yield from Query.__reversed__(source)

    def reverse(self: Iterable[T]) -> "Query[T]":
        return Query(Query._reverse(self))

    @staticmethod
    @lazy
    def _enumerate(source: Iterable[T], start: int = 0) -> Iterator[Tuple[int, T]]:
        yield from enumerate(source, start)

    def enumerate(self: Iterable[T], start: int = 0) -> "Query[Tuple[int, T]]":
        return Query(Query._enumerate(self, start))

    @staticmethod
    @lazy
    def _filter(source, func):
        return filter(func, source)

    def filter(self, func):
        return Query(Query._filter(self, func))

    @staticmethod
    @lazy
    def _map(source: Iterable[T], selector: Callable[[T], R]) -> Iterator[R]:
        return map(selector, source)

    @staticmethod
    @lazy
    def _map_unpack(source: Iterable[T], selector: Callable[..., R]) -> Iterator[R]:
        for elm in source:
            yield selector(*elm)  # type: ignore

    @staticmethod
    @lazy
    def _map_unpack_kw(source: Iterable[T], selector: Callable[..., R]) -> Iterator[R]:
        for elm in source:
            yield selector(**elm)  # type: ignore

    def map(self, selector: Callable[[T], R]) -> "Query[R]":
        return Query(Query._map(self, selector))

    def map_unpack(self, selector: Callable[..., R]) -> "Query[R]":
        return Query(Query._map_unpack(self, selector))

    def map_unpack_kw(self, selector: Callable[..., R]) -> "Query[R]":
        return Query(Query._map_unpack_kw(self, selector))

    @staticmethod
    @lazy
    def _slice(source: Iterable[T], start, end) -> Iterator[T]:
        if start < 0:
            start = 0

        it = iter(source)

        current = 0

        try:
            while current < start:
                next(it)
                current += 1

            while current < end:
                yield next(it)
                current += 1

        except StopIteration:
            pass

    def slice(self, start: int = 0, end: int = None) -> "Query[T]":
        if start == 0 and end is None:
            return self

        end = float("inf") if end is None else end  # type: ignore
        return Query(Query._slice(self, start, end))

    def save(self):
        return Query(list(self))

    def to_list(self):
        return list(self)

    def to_dict(self: Iterable[Tuple[K, V]]):
        return dict(iter(self))

    def to_index(self: Iterable[Tuple[K, V]]) -> "QuerableDict[K, V]":
        return QuerableDict(self)

    def to_lookup(
        self, selector: Callable[[T], R] = None
    ) -> "QuerableDict[R, Iterable[T]]":
        # 指定したキーを集約する
        raise NotImplementedError()

    def grouping(self, selector: Callable[[T], R]):
        # to_lookupとあまり変わらないが
        # to_lookupは即時実行groupingが遅延評価
        raise NotImplementedError()


class QueryTuple(Query[Tuple[K, V]], Iterable[Tuple[K, V]], Generic[K, V]):
    pass


class QuerableDict(QueryTuple[K, V]):
    @overload
    def __init__(self, source: Mapping[K, V]):
        pass

    @overload
    def __init__(self, source: Iterable[Tuple[K, V]]):
        pass

    def __init__(self, source):
        if isinstance(source, Mapping):
            self.source = source
        else:
            self.source = {k: v for k, v in source}

    def __iter__(self) -> Iterator[Tuple[K, V]]:
        return self.source.items().__iter__()

    def len(self):
        return len(self.source)

    def save(self):
        raise NotImplementedError()

    def to_index(self):
        raise NotImplementedError()

    def keys(self) -> Query[K]:
        return Query(self.source.keys())

    def values(self) -> Query[V]:
        return Query(self.source.values())

    def items(self) -> Query[Tuple[K, V]]:
        return self

    def __getitem__(self, key):
        return self.source[key]  # type: ignore

    def get(self, key) -> V:
        return self[key]

    def get_or_none(self, key) -> Union[V, None]:
        return self.get_or_default(key, None)

    def get_or_default(self, key, default: R = None) -> Union[V, R, None]:
        try:
            return self[key]
        except KeyError:
            return default

    def get_many(self, *keys, duplicate: bool = True) -> Query[Tuple[K, V]]:
        if not duplicate:
            keys = set(keys)  # type: ignore
        return Query(QuerableDict._get_many(self, keys))

    def get_many_or_raise(self, *keys, duplicate: bool = True) -> Query[Tuple[K, V]]:
        raise NotImplementedError()

    @staticmethod
    @lazy
    def _get_many(source: "QuerableDict", keys: Iterable) -> Iterator:
        for id in keys:
            obj = source.get_or_default(id, undefined)
            if not obj is undefined:
                yield id, obj


@overload
def pnq(source: Mapping[K, V]) -> QuerableDict[K, V]:
    ...


@overload
def pnq(source: Iterable[Tuple[K, V]]) -> QueryTuple[K, V]:
    ...


@overload
def pnq(source: Iterable[T]) -> Query[T]:
    ...


def pnq(source):
    if isinstance(source, Mapping):
        return QuerableDict(source)
    else:
        return Query(source)


KT = TypeVar("KT", bound=Any)


def get_first(source: Iterable[Tuple[KT, V]]) -> Tuple[KT, V]:
    ...


# TODO: Tuple[Literal, Literal]が返ってきてしまう
a = get_first([(1, 2)])
