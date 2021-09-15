# type: ignore

from functools import wraps
from operator import attrgetter, itemgetter
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterable,
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
    overload,
)

from . import actions
from .core import LazyIterate as _LazyIterate
from .core import LazyReference as _LazyReference
from .core import piter, undefined
from .exceptions import NoElementError, NotOneElementError
from .op import TH_ASSIGN_OP

T = TypeVar("T")
K = TypeVar("K")
V = TypeVar("V")
K2 = TypeVar("K2")
V2 = TypeVar("V2")
R = TypeVar("R")


__all__ = ["Query", "PairQuery", "IndexQuery", "ListEx", "DictEx", "SetEx", "query", "undefined"]


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


class Query(Generic[T], Iterable[T]):
    def to_list(self) -> "ListEx[T]":
        pass

    def to_dict(self) -> "DictEx[Any, Any]":
        pass


class PairQuery(Generic[K, V], Iterable[Tuple[K, V]]):
    def to_list(self) -> "ListEx[Tuple[K, V]]":
        pass

    def to_dict(self) -> "DictEx[K, V]":
        pass


class IndexQuery(Generic[K, V]):
    pass


class ListEx(Generic[T], Sequence[T]):
    pass


class DictEx(Generic[K, V], Mapping[K, V]):
    pass


class SetEx(Generic[T], Iterable[T]):
    pass

{% for query in queries %}


    {% if not query.is_index %}


class {{query.cls}}:
    def len(self) -> int:
        return actions.len(self)

    def exists(self) -> bool:
        return actions.exists(self)

    def all(self, selector: Callable[[{{query.row}}], Any]=lambda x: x) -> bool:
        return actions.all(self, selector)

    def any(self, selector: Callable[[{{query.row}}], Any]=lambda x: x) -> bool:
        return actions.any(self, selector)

    def contains(self, value, selector: Callable[[{{query.row}}], Any]=lambda x: x) -> bool:
        return actions.contains(self, value, selector)

    @overload
    def min(self, *, default=NoReturn) -> Union[{{query.value}}, NoReturn]: ...

    @overload
    def min(self, selector: Callable[[{{query.row}}], R]=lambda x: x, default=NoReturn) -> R: ...
    def min(self, selector: Callable[[{{query.row}}], R]=lambda x: x, default=NoReturn) -> R:
        return actions.min(self, selector, default)

    @overload
    def max(self, *, default=NoReturn) -> Union[{{query.value}}, NoReturn]: ...
    @overload
    def max(self, selector: Callable[[{{query.row}}], R]=lambda x: x, default=NoReturn) -> R: ...
    def max(self, selector: Callable[[{{query.row}}], R]=lambda x: x, default=NoReturn) -> R:
        return actions.max(self, selector, default)

    @overload
    def sum(self) -> {{query.value}}: ...
    @overload
    def sum(self, selector: Callable[[{{query.row}}], R]=lambda x: x) -> R: ...
    def sum(self, selector: Callable[[{{query.row}}], R]=lambda x: x) -> R:
        return actions.sum(self, selector)

    @overload
    def average(self) -> {{query.value}}: ...
    @overload
    def average(self, selector: Callable[[{{query.row}}], R]=lambda x: x) -> R: ...
    def average(self, selector: Callable[[{{query.row}}], R]=lambda x: x) -> R:
        return actions.average(self, selector)

    def reduce(self, seed: T, op: Union[TH_ASSIGN_OP, Callable[[Any, Any], Any]] = "+=", selector=lambda x: x) -> T:
        return actions.reduce(self, seed, op, selector)

    def concat(self, selector=lambda x: x, delimiter: str = "") -> str:
        return actions.concat(self, selector, delimiter)

    def dispatch(self, func: Callable, selector: Callable[[{{query.row}}], Any], on_error: Callable=...) -> Tuple[int, int]: ...
    def to_list(self) -> {{query.to_list}}:
        return ListEx(piter(self))
    def to_dict(self, duplicate: bool=...) -> {{query.to_dict}}:
        return DictEx(piter(self))

    @overload
    def one(self) -> {{query.row}}: ...
    @overload
    def one(self, selector: Callable[[{{query.row}}], R]={{query.selector}}) -> R: ...
    def one(self, selector: Callable[[{{query.row}}], R]={{query.selector}}) -> R:
        it = piter(self)
        try:
            result = next(it)
        except StopIteration:
            raise NoElementError()

        try:
            next(it)
            raise NotOneElementError()
        except StopIteration:
            pass

        return selector(result)
    @overload
    def first(self) -> {{query.row}}: ...
    @overload
    def first(self, selector: Callable[[{{query.row}}], R]={{query.selector}}) -> R: ...
    def first(self, selector: Callable[[{{query.row}}], R]={{query.selector}}) -> Any:
        if isinstance(self, Sequence):
            try:
                obj = self[0]
            except IndexError:
                raise NoElementError()
        else:
            try:
                it = piter(self)
                obj = next(it)
            except StopIteration:
                raise NoElementError()

        return selector(obj)

    @overload
    def last(self) -> {{query.row}}: ...
    @overload
    def last(self, selector: Callable[[{{query.row}}], R]={{query.selector}}) -> R: ...
    def last(self, selector: Callable[[{{query.row}}], R]={{query.selector}}) -> R:
        undefined = object()
        last: R = undefined  # type: ignore
        for elm in piter(self):
            last = elm  # type: ignore

        if last is undefined:
            raise NoElementError()

        return selector(last)  # type: ignore
    @overload
    def one_or_default(self) -> Union[{{query.row}}, None]: ...
    @overload
    def one_or_default(self, default: R) -> Union[{{query.row}}, R]: ...
    def one_or_default(self, default=None) -> Any:
        try:
            return self.one()
        except (NoElementError, NotOneElementError):
            return default
    @overload
    def first_or_default(self) -> Union[{{query.row}}, None]: ...
    @overload
    def first_or_default(self, default: R) -> Union[{{query.row}}, R]: ...
    def first_or_default(self, default=None) -> Any:
        try:
            return self.first()
        except NoElementError:
            return default
    @overload
    def last_or_default(self) -> Union[{{query.row}}, None]: ...
    @overload
    def last_or_default(self, default: R) -> Union[{{query.row}}, R]: ...
    def last_or_default(self, default=None) -> Any:
        try:
            return self.last()
        except NoElementError:
            return default

    @overload
    def cast(self, type: Type[Tuple[K2, V2]]) -> {{pair.name}}[K2, V2]:
        pass

    @overload
    def cast(self, type: Type[R]) -> {{sequence.name}}[R]:
        pass


    @overload
    def cast(self, type: Callable[[{{query.row}}], Tuple[K2, V2]]) -> {{pair.name}}[K2, V2]:
        pass

    @overload
    def cast(self, type: Callable[[{{query.row}}], R]) -> {{sequence.name}}[R]:
        pass


    def cast(self, type: Type[R]) -> {{sequence.name}}[R]:
        return self

    @lazy_iterate
    def enumerate(self, start: int = 0, step: int = 1) -> {{pair.name}}[int, {{query.row}}]:
        yield from (x * step for x in enumerate(self, start))

    @overload
    def map(self, type: Callable[[{{query.row}}], Tuple[K2, V2]]) -> {{pair.name}}[K2, V2]:
        pass

    @overload
    def map(self, type: Callable[[{{query.row}}], R]) -> {{sequence.name}}[R]:
        pass

    @lazy_iterate
    def map(self, selector: Callable[[{{query.row}}], R]) -> {{sequence.name}}[R]:
        if selector is str:
            selector = lambda x: "" if x is None else str(x)
        yield from map(selector, self)

    {% if query.is_pair %}

    @overload
    def select(self, item: Literal[0]) -> "Query[{{query.K}}]":
        ...

    @overload
    def select(self, item: Literal[1]) -> "Query[{{query.V}}]":
        ...

    {% endif %}

    @overload
    def select(self, item) -> "Query[Any]":
        ...

    @overload
    def select(self, item, *items) -> "Query[Tuple]":
        ...

    @lazy_iterate
    def select(self, *items) -> "Query[Any]":
        selector = itemgetter(*items)
        yield from map(lambda x: selector(x), self)

    select_item = select

    @overload
    def select_attr(self, attr: str) -> "Query[Any]":
        ...

    @overload
    def select_attr(self, attr: str, *attrs: str) -> "Query[Tuple]":
        ...

    @lazy_iterate
    def select_attr(self, *attrs: str) -> "Query[Any]":
        selector = attrgetter(*attrs)
        yield from map(lambda x: selector(x), self)

    def select_items(self, *items) -> "Query[Tuple]":
        if len(items) == 0:
            selector = lambda x: tuple()  # type: ignore
        elif len(items) == 1:
            # itemgetter/getattrは引数が１の時、タプルでなくそのセレクトされた値を直接返のでタプルで返すようにする
            name = items[0]
            selector = lambda x: (x[name],)
        else:
            selector = itemgetter(*items)

        def pmap(self, selector):
            yield from map(selector, self)

        return LazyIterate(pmap, self, selector)

    def select_attrs(self, *attrs: Any) -> "Query[Tuple]":
        if len(attrs) == 0:
            selector = lambda x: tuple()  # type: ignore
        elif len(attrs) == 1:
            # itemgetter/getattrは引数が１の時、タプルでなくそのセレクトされた値を直接返のでタプルで返すようにする
            name = attrs[0]
            selector = lambda x: (getattr(x, name),)
        else:
            selector = attrgetter(*attrs)

        def pmap(self, selector):
            yield from map(selector, self)

        return LazyIterate(pmap, self, selector)

    @lazy_iterate
    def unpack(self, selector: Callable[..., R]) -> {{sequence.name}}[R]:
        for elm in self:
            yield selector(*elm)  # type: ignore

    @lazy_iterate
    def unpack_kw(self, selector: Callable[..., R]) -> {{sequence.name}}[R]:
        for elm in self:
            yield selector(**elm)  # type: ignore

    @lazy_iterate
    def group_by(self, selector: Callable[[{{query.row}}], Tuple[K2, V2]]) -> {{pair.name}}[K2, List[V2]]:
        results: Dict[K, List[V]] = defaultdict(list)
        for elm in self:
            k, v = selector(elm)
            results[k].append(v)

        for k, v in results.items():  # type: ignore
            yield k, v

    def join(self, right, on: Callable[[Tuple[list, list]], Callable], select):
        [].join(
            [],
            lambda left, right: left.name == right.name,
            lambda left, right: (left.name, right.age)
        )

        table(User).join(
            Item,
            on=User.id == Item.id
        ).select(User.id, Item.id)

        pass

    @lazy_iterate
    def distinct(self, selector: Callable[[{{query.row}}], Any], msg: str=...) -> {{query.str}}:
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
    def filter(self, predicate: Callable[[{{query.row}}], bool]) -> {{query.str}}:
        yield from filter(predicate, self)  # type: ignore

    @lazy_iterate
    def filter_type(self, *types: Type[R]) -> {{query.str}}:
        def normalize(type):
            if type is None:
                return None.__class__
            else:
                return type

        types = tuple((normalize(x) for x in types))
        return filter(lambda x: isinstance(x, *types), self)


    @lazy_iterate
    def must(self, predicate: Callable[[{{query.row}}], bool], msg: str=...) -> {{query.str}}:
        """要素の検証に失敗した時例外を発生させる。"""
        for elm in self:
            if not predicate(elm):
                raise ValueError(f"{msg} {elm}")
            yield elm

    @lazy_iterate
    def must_unique(self, selector: Callable[[T], R]):
        seen = set()
        duplicated = []
        for elm in source:
            if selector(elm) in seen:
                duplicated.append(elm)
            else:
                seen.add(selector(elm))

        if duplicated:
            raise ValueError(f"Duplicated elements: {duplicated}")

        for elm in source:
            yield elm




    @lazy_iterate
    def skip(self, count: int) -> {{query.str}}:
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
    def take(self, count: int) -> {{query.str}}:
        current = 0

        try:
            while current < count:
                yield next(self)
                current += 1
        except StopIteration:
            return

    @lazy_iterate
    def range(self, start: int=..., stop: int=...) -> {{query.str}}:
        if start < 0:
            start = 0

        if stop is None:
            stop = float("inf")
        elif stop < 0:
            stop = 0
        else:
            pass

        current = 0

        try:
            while current < start:
                next(self)
                current += 1
        except StopIteration:
            return

        try:
            while current < stop:
                yield next(self)
                current += 1
        except StopIteration:
            return

    @lazy_reference
    def page(self, page: int=..., size: int=...) -> {{query.str}}:
        start, stop = page_calc(page, size)
        yield from self.range(start, stop)


    @lazy_reference
    def reverse(self) -> {{query.str}}:
        yield actions.reverse(self)

    @lazy_iterate
    def order(self, selector, desc: bool = False) -> {{query.str}}:
        yield from sorted(self, key=selector, reverse=desc)

    def order_by_items(
        self, *items: Any, desc: bool = False
    ) -> {{query.str}}:
        selector = itemgetter(*items)
        return self.order(selector, desc)

    def order_by_attrs(
        self: Iterable[T], *attrs: str, desc: bool = False
    ) -> {{query.str}}:
        selector = attrgetter(*attrs)
        return self.order(selector, desc)

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

    {% else %}
class {{query.cls}}:

    @lazy_reference
    def get_many(self, *keys: {{query.K}}) -> {{query.str}}:
        undefined = object()
        for id in keys:
            obj = self.get(id, undefined)
            if not obj is undefined:
                yield id, obj
    @overload
    def get(self, key: {{query.K}}) -> {{query.V}}: ...
    @overload
    def get(self, key: {{query.K}}, default: R=...) -> Union[{{query.V}}, R]: ...
    def get(self, key: {{query.K}}, default: R=undefined) -> Any:
        try:
            return self[key]  # type: ignore
        except (KeyError, IndexError):
            if default is not undefined:
                return default
            else:
                raise

    def get_or_default(self, key,default=None):
        return self.get(key, default)  # type: ignore

    def get_or_none(self, key):
        return self.get(key, None)  # type: ignore

    def to_list(self) -> {{query.to_list}}:
        return ListEx(piter(self))

    def to_dict(self) -> {{query.to_dict}}:
        return DictEx(piter(self))

    {% endif %}
{% endfor %}



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


class SetEx(Instance, IndexQuery[T, T], Query[T], Set[T]):
    def __piter__(self):
        return self.__iter__()

    def __getitem__(self, key: T):
        if key in self:
            return key
        else:
            raise KeyError(key)

    @lazy_reference
    def reverse(self) -> "Query[T]":
        yield from reversed(self)


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
    elif hasattr(source, "__iter__"):
        return LazyIterate(iter, source)
    else:
        raise Exception()


def repeat(func, *args, **kwargs):
    def iterate():
        while True:
            yield func(*args, **kwargs)

    return LazyIterate(iterate())


def count(start=0, step=1):
    from itertools import count
    return LazyIterate(count(start, step))


def cycle(iterable, repeat=None):
    from itertools import cycle
    return LazyIterate(cycle(iterable))


def page_calc(page: int, size: int):
    if size < 0:
        raise ValueError("size must be >= 0")
    start = (page - 1) * size
    stop = start + size
    return start, stop
