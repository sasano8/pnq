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

    def one(self) -> T:
        it = iter(self.source)

        obj = next(it)

        is_one = True

        try:
            obj_next = next(it)
            is_one = False
        except:
            pass

        if is_one:
            raise Exception("not one.")

        return obj

    def one_or_none(self) -> Union[T, None]:
        try:
            return self.one()
        except StopIteration:
            return None

    @staticmethod
    @lazy
    def _order(
        source: Iterable[T], fields: set = None, desc: bool = False, default_fields=None
    ) -> Iterator[T]:
        if fields is None:
            fields = default_fields

        if fields is None:
            sorter = None
        else:
            if len(fields) == 1:
                field = fields[0]
                sorter = lambda x: getattr(x, field)
            else:
                sorter = lambda x: tuple((getattr(x, field) for field in fields))

        yield from sorted(source, key=sorter, reverse=desc)

    def order(
        self: Iterable[T], fields: tuple = None, desc: bool = False
    ) -> "Query[T]":
        return Query(Query._order(self, fields, desc))

    @staticmethod
    @lazy
    def _enumerate(source: Iterable[T], start: int = 0) -> Iterator[Tuple[int, T]]:
        yield from enumerate(source, start)

    def enumrate(self: Iterable[T], start: int = 0) -> "Query[Tuple[int, T]]":
        return Query(Query._enumerate(self, start))

    @staticmethod
    @lazy
    def _filter(source, func):
        return filter(func, source)

    def filter(self, func):
        return Query(Query._filter(self, func))

    @staticmethod
    @lazy
    def _map(source, func: Callable[[T], R]) -> Iterator[R]:
        return map(func, source)

    def map(self, func: Callable[[T], R]) -> "Query[R]":
        return Query(Query._map(self, func))

    @staticmethod
    @lazy
    def _lookup(source, func: Callable[[T], R]):
        return map(lambda x: (func(x), x), source)

    def lookup(self, func: Callable[[T], K]) -> "Query[Tuple[K, T]]":
        return Query(Query._lookup(self, func))

    def save(self):
        return Query(list(self))

    def to_list(self):
        return list(self)

    def to_dict(self: Iterable[Tuple[K, V]]):
        return dict(iter(self))

    def to_index(self: Iterable[Tuple[K, V]]) -> "QuerableDict[K, V]":
        return QuerableDict(self)


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

    # def get(self, id) -> V:
    #     return self.source[id]

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
