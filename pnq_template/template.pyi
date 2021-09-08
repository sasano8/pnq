# type: ignore

from functools import wraps
from operator import attrgetter, itemgetter
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Generic,
    Iterable,
    List,
    Mapping,
    Sequence,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
    overload,
)

from .core import LazyIterate as _LazyIterate
from .core import LazyReference as _LazyReference
from .core import piter, undefined
from .exceptions import NoElementError, NotOneError

T = TypeVar("T")
K = TypeVar("K")
V = TypeVar("V")
K2 = TypeVar("K2")
V2 = TypeVar("V2")
R = TypeVar("R")


__all__ = ["Query", "PairQuery", "IndexQuery", "ListEx", "DictEx", "SetEx"]


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


class Query(Generic[T]):
    pass


class PairQuery(Generic[K, V]):
    pass


class IndexQuery(Generic[K, V]):
    pass


class ListEx(Generic[T]):
    pass


class DictEx(Generic[K, V]):
    pass


class SetEx(Generic[K, V]):
    pass

{% for query in queries %}


    {% if not query.is_index %}


class {{query.cls}}:
    def len(self) -> int:
        return len(self)

    def exists(self) -> bool:
        return len(self) > 0

    def all(self, selector: Callable[[{{query.row}}], Any]={{query.selector}}) -> bool:
        return all(map(selector, piter(self)))

    def any(self, selector: Callable[[{{query.row}}], Any]={{query.selector}}) -> bool:
        return any(map(selector, piter(self)))

    @overload
    def min(self) -> {{query.value}}: ...

    @overload
    def min(self, selector: Callable[[{{query.row}}], R]={{query.selector}}) -> R: ...
    def min(self, selector: Callable[[{{query.row}}], R]={{query.selector}}) -> R:
        return min(map(selector, piter(self)))

    @overload
    def max(self) -> {{query.value}}: ...
    @overload
    def max(self, selector: Callable[[{{query.row}}], R]={{query.selector}}) -> R: ...
    def max(self, selector: Callable[[{{query.row}}], R]={{query.selector}}) -> R:
        return max(map(selector, piter(self)))

    @overload
    def sum(self) -> {{query.value}}: ...
    @overload
    def sum(self, selector: Callable[[{{query.row}}], R]={{query.selector}}) -> R: ...
    def sum(self, selector: Callable[[{{query.row}}], R]={{query.selector}}) -> R:
        return sum(map(selector, piter(self)))

    @overload
    def average(self) -> {{query.value}}: ...
    def average(self, selector: Callable[[{{query.row}}], R]={{query.selector}}) -> R: ...
    @overload
    def average(self, selector: Callable[[{{query.row}}], R]={{query.selector}}) -> R:
        import statistics
        return statistics.mean(pmap(self, selector))  # type: ignore

    def reduce(self, accumulator: Callable[[T, T], T], seed: T=undefined) -> Any:
        from functools import reduce

        if seed is undefined:
            return reduce(accumulator, self)
        else:
            return reduce(accumulator, self, seed)
    def dispatch(self, func: Callable, selector: Callable[[{{query.row}}], Any], on_error: Callable=...) -> Tuple[int, int]: ...
    def to_list(self) -> {{query.to_list}}:
        return ListEx(piter(self))
    def to_dict(self, duplicate: bool=...) -> {{query.to_dict}}:
        return DictEx(self)

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
            raise NotOneError()
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
        except (NoElementError, NotOneError):
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

    def cast(self, type: Type[R]) -> {{sequence.name}}[R]:
        return self

    @lazy_iterate
    def enumerate(self, start: int=0) -> {{pair.name}}[int, {{query.row}}]:
        yield from enumerate(self, start)

    @lazy_iterate
    def map(self, selector: Callable[[{{query.row}}], R]) -> {{sequence.name}}[R]:
        if selector is str:
            selector = lambda x: "" if x is None else str(x)
        yield from map(selector, self)

    @lazy_iterate
    def pairs(self, selector: Callable[[{{query.row}}], Tuple[K, V]]) -> {{pair.name}}[K2, V2]:
        yield from map(selector, self)

    @lazy_iterate
    def select(self, item) -> "Query[Any]":
        yield from map(lambda x: x[item], self)

    @lazy_iterate
    def select_item(self, item) -> "Query[Any]":
        yield from map(lambda x: x[item], self)

    @lazy_iterate
    def select_attr(self, attr: str) -> "Query[Any]":
        yield from map(lambda x: getattr(x, attr), self)

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


    def select_items(self, *items: str) -> "Query[Tuple]":
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
    def unique(self, selector: Callable[[{{query.row}}], Any], msg: str=...) -> {{query.str}}:
        duplicate = set()
        for elm in self:
            value = selector(elm)
            if not value in duplicate:
                duplicate.add(value)
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
        def _page_calc(page: int, size: int):
            if size < 0:
                raise ValueError("size must be >= 0")
            start = (page - 1) * size
            stop = start + size
            return start, stop

        start, stop = _page_calc(page, size)
        yield from self.range(start, stop)

    @lazy_reference
    def reverse(self) -> {{query.str}}:
        if hasattr(self, "__reversed__"):
            if isinstance(self, Mapping):
                yield from ((k,self[k]) for k in self.__reversed__())
            else:
                yield from self.__reversed__()
        else:
            yield from list(piter(self)).__reversed__()

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


class ListEx(IndexQuery[int, T], Query[T], List[T]):
    def __piter__(self):
        return self.__iter__()

    @property
    def source(self):
        return self


class DictEx(IndexQuery[K, V], PairQuery[K, V], Dict[K, V]):
    def __piter__(self):
        return self.items().__iter__()

    @property
    def source(self):
        return self

    @lazy_reference
    def keys(self):
        yield from super().keys()

    @lazy_reference
    def values(self):
        yield from super().values()

    @lazy_reference
    def items(self):
        yield from super().items()

class SetEx(IndexQuery[T, T], Query[T], Set[T]):
    def __piter__(self):
        return self.__iter__()

    @property
    def source(self):
        return self

    def __getitem__(self, key: T):
        if key in self:
            return key
        else:
            raise KeyError(key)


@overload
def query(source: Mapping[K, V]) -> DictEx[K, V]:
    ...


@overload
def query(source: Iterable[T]) -> ListEx[T]:
    ...


def query(source: T) -> T:
    if hasattr(source, "__piter__"):
        return source
    elif isinstance(source, dict):
        return DictEx(source)
    elif isinstance(source, list):
        return ListEx(source)
    else:
        raise Exception()
