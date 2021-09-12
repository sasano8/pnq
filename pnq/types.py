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
from .exceptions import NoElementError, NotOneError

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


class Query(Generic[T]):
    def len(self) -> int:
        return len(self)

    def exists(self) -> bool:
        return len(self) > 0

    def all(self, selector: Callable[[T], Any] = lambda x: x) -> bool:
        return all(map(selector, piter(self)))

    def any(self, selector: Callable[[T], Any] = lambda x: x) -> bool:
        return any(map(selector, piter(self)))

    @overload
    def min(self) -> T:
        ...

    @overload
    def min(self, selector: Callable[[T], R] = lambda x: x) -> R:
        ...

    def min(self, selector: Callable[[T], R] = lambda x: x) -> R:
        return min(map(selector, piter(self)))

    @overload
    def max(self) -> T:
        ...

    @overload
    def max(self, selector: Callable[[T], R] = lambda x: x) -> R:
        ...

    def max(self, selector: Callable[[T], R] = lambda x: x) -> R:
        return max(map(selector, piter(self)))

    @overload
    def sum(self) -> T:
        ...

    @overload
    def sum(self, selector: Callable[[T], R] = lambda x: x) -> R:
        ...

    def sum(self, selector: Callable[[T], R] = lambda x: x) -> R:
        return sum(map(selector, piter(self)))

    @overload
    def average(self) -> T:
        ...

    def average(self, selector: Callable[[T], R] = lambda x: x) -> R:
        ...

    @overload
    def average(self, selector: Callable[[T], R] = lambda x: x) -> R:
        import statistics

        return statistics.mean(pmap(self, selector))  # type: ignore

    def reduce(self, accumulator: Callable[[T, T], T], seed: T = undefined) -> Any:
        from functools import reduce

        if seed is undefined:
            return reduce(accumulator, self)
        else:
            return reduce(accumulator, self, seed)

    def dispatch(
        self, func: Callable, selector: Callable[[T], Any], on_error: Callable = ...
    ) -> Tuple[int, int]:
        ...

    def to_list(self) -> ListEx[T]:
        return ListEx(piter(self))

    def to_dict(self, duplicate: bool = ...) -> DictEx[Any, Any]:
        return DictEx(piter(self))

    @overload
    def one(self) -> T:
        ...

    @overload
    def one(self, selector: Callable[[T], R] = lambda x: x) -> R:
        ...

    def one(self, selector: Callable[[T], R] = lambda x: x) -> R:
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
    def first(self) -> T:
        ...

    @overload
    def first(self, selector: Callable[[T], R] = lambda x: x) -> R:
        ...

    def first(self, selector: Callable[[T], R] = lambda x: x) -> Any:
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
    def last(self) -> T:
        ...

    @overload
    def last(self, selector: Callable[[T], R] = lambda x: x) -> R:
        ...

    def last(self, selector: Callable[[T], R] = lambda x: x) -> R:
        undefined = object()
        last: R = undefined  # type: ignore
        for elm in piter(self):
            last = elm  # type: ignore

        if last is undefined:
            raise NoElementError()

        return selector(last)  # type: ignore

    @overload
    def one_or_default(self) -> Union[T, None]:
        ...

    @overload
    def one_or_default(self, default: R) -> Union[T, R]:
        ...

    def one_or_default(self, default=None) -> Any:
        try:
            return self.one()
        except (NoElementError, NotOneError):
            return default

    @overload
    def first_or_default(self) -> Union[T, None]:
        ...

    @overload
    def first_or_default(self, default: R) -> Union[T, R]:
        ...

    def first_or_default(self, default=None) -> Any:
        try:
            return self.first()
        except NoElementError:
            return default

    @overload
    def last_or_default(self) -> Union[T, None]:
        ...

    @overload
    def last_or_default(self, default: R) -> Union[T, R]:
        ...

    def last_or_default(self, default=None) -> Any:
        try:
            return self.last()
        except NoElementError:
            return default

    @overload
    def cast(self, type: Type[Tuple[K2, V2]]) -> PairQuery[K2, V2]:
        pass

    @overload
    def cast(self, type: Type[R]) -> Query[R]:
        pass

    @overload
    def cast(self, type: Callable[[T], Tuple[K2, V2]]) -> PairQuery[K2, V2]:
        pass

    @overload
    def cast(self, type: Callable[[T], R]) -> Query[R]:
        pass

    def cast(self, type: Type[R]) -> Query[R]:
        return self

    @lazy_iterate
    def enumerate(self, start: int = 0, step: int = 1) -> PairQuery[int, T]:
        yield from (x * step for x in enumerate(self, start))

    @overload
    def map(self, type: Callable[[T], Tuple[K2, V2]]) -> PairQuery[K2, V2]:
        pass

    @overload
    def map(self, type: Callable[[T], R]) -> Query[R]:
        pass

    @lazy_iterate
    def map(self, selector: Callable[[T], R]) -> Query[R]:
        if selector is str:
            selector = lambda x: "" if x is None else str(x)
        yield from map(selector, self)

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
    def unpack(self, selector: Callable[..., R]) -> Query[R]:
        for elm in self:
            yield selector(*elm)  # type: ignore

    @lazy_iterate
    def unpack_kw(self, selector: Callable[..., R]) -> Query[R]:
        for elm in self:
            yield selector(**elm)  # type: ignore

    @lazy_iterate
    def group_by(
        self, selector: Callable[[T], Tuple[K2, V2]]
    ) -> PairQuery[K2, List[V2]]:
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
            lambda left, right: (left.name, right.age),
        )

        table(User).join(Item, on=User.id == Item.id).select(User.id, Item.id)

        pass

    @lazy_iterate
    def distinct(self, selector: Callable[[T], Any], msg: str = ...) -> Query[T]:
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
    def filter(self, predicate: Callable[[T], bool]) -> Query[T]:
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

    @lazy_iterate
    def must(self, predicate: Callable[[T], bool], msg: str = ...) -> Query[T]:
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
    def skip(self, count: int) -> Query[T]:
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
    def take(self, count: int) -> Query[T]:
        current = 0

        try:
            while current < count:
                yield next(self)
                current += 1
        except StopIteration:
            return

    @lazy_iterate
    def range(self, start: int = ..., stop: int = ...) -> Query[T]:
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
    def page(self, page: int = ..., size: int = ...) -> Query[T]:
        start, stop = page_calc(page, size)
        yield from self.range(start, stop)

    @lazy_reference
    def reverse(self) -> Query[T]:
        yield actions.reverse(self)

    @lazy_iterate
    def order(self, selector, desc: bool = False) -> Query[T]:
        yield from sorted(self, key=selector, reverse=desc)

    def order_by_items(self, *items: Any, desc: bool = False) -> Query[T]:
        selector = itemgetter(*items)
        return self.order(selector, desc)

    def order_by_attrs(self: Iterable[T], *attrs: str, desc: bool = False) -> Query[T]:
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


class PairQuery(Generic[K, V]):
    def len(self) -> int:
        return len(self)

    def exists(self) -> bool:
        return len(self) > 0

    def all(self, selector: Callable[[Tuple[K, V]], Any] = lambda x: x[0]) -> bool:
        return all(map(selector, piter(self)))

    def any(self, selector: Callable[[Tuple[K, V]], Any] = lambda x: x[0]) -> bool:
        return any(map(selector, piter(self)))

    @overload
    def min(self) -> V:
        ...

    @overload
    def min(self, selector: Callable[[Tuple[K, V]], R] = lambda x: x[0]) -> R:
        ...

    def min(self, selector: Callable[[Tuple[K, V]], R] = lambda x: x[0]) -> R:
        return min(map(selector, piter(self)))

    @overload
    def max(self) -> V:
        ...

    @overload
    def max(self, selector: Callable[[Tuple[K, V]], R] = lambda x: x[0]) -> R:
        ...

    def max(self, selector: Callable[[Tuple[K, V]], R] = lambda x: x[0]) -> R:
        return max(map(selector, piter(self)))

    @overload
    def sum(self) -> V:
        ...

    @overload
    def sum(self, selector: Callable[[Tuple[K, V]], R] = lambda x: x[0]) -> R:
        ...

    def sum(self, selector: Callable[[Tuple[K, V]], R] = lambda x: x[0]) -> R:
        return sum(map(selector, piter(self)))

    @overload
    def average(self) -> V:
        ...

    def average(self, selector: Callable[[Tuple[K, V]], R] = lambda x: x[0]) -> R:
        ...

    @overload
    def average(self, selector: Callable[[Tuple[K, V]], R] = lambda x: x[0]) -> R:
        import statistics

        return statistics.mean(pmap(self, selector))  # type: ignore

    def reduce(self, accumulator: Callable[[T, T], T], seed: T = undefined) -> Any:
        from functools import reduce

        if seed is undefined:
            return reduce(accumulator, self)
        else:
            return reduce(accumulator, self, seed)

    def dispatch(
        self,
        func: Callable,
        selector: Callable[[Tuple[K, V]], Any],
        on_error: Callable = ...,
    ) -> Tuple[int, int]:
        ...

    def to_list(self) -> ListEx[Tuple[K, V]]:
        return ListEx(piter(self))

    def to_dict(self, duplicate: bool = ...) -> DictEx[K, V]:
        return DictEx(piter(self))

    @overload
    def one(self) -> Tuple[K, V]:
        ...

    @overload
    def one(self, selector: Callable[[Tuple[K, V]], R] = lambda x: x[0]) -> R:
        ...

    def one(self, selector: Callable[[Tuple[K, V]], R] = lambda x: x[0]) -> R:
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
    def first(self) -> Tuple[K, V]:
        ...

    @overload
    def first(self, selector: Callable[[Tuple[K, V]], R] = lambda x: x[0]) -> R:
        ...

    def first(self, selector: Callable[[Tuple[K, V]], R] = lambda x: x[0]) -> Any:
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
    def last(self) -> Tuple[K, V]:
        ...

    @overload
    def last(self, selector: Callable[[Tuple[K, V]], R] = lambda x: x[0]) -> R:
        ...

    def last(self, selector: Callable[[Tuple[K, V]], R] = lambda x: x[0]) -> R:
        undefined = object()
        last: R = undefined  # type: ignore
        for elm in piter(self):
            last = elm  # type: ignore

        if last is undefined:
            raise NoElementError()

        return selector(last)  # type: ignore

    @overload
    def one_or_default(self) -> Union[Tuple[K, V], None]:
        ...

    @overload
    def one_or_default(self, default: R) -> Union[Tuple[K, V], R]:
        ...

    def one_or_default(self, default=None) -> Any:
        try:
            return self.one()
        except (NoElementError, NotOneError):
            return default

    @overload
    def first_or_default(self) -> Union[Tuple[K, V], None]:
        ...

    @overload
    def first_or_default(self, default: R) -> Union[Tuple[K, V], R]:
        ...

    def first_or_default(self, default=None) -> Any:
        try:
            return self.first()
        except NoElementError:
            return default

    @overload
    def last_or_default(self) -> Union[Tuple[K, V], None]:
        ...

    @overload
    def last_or_default(self, default: R) -> Union[Tuple[K, V], R]:
        ...

    def last_or_default(self, default=None) -> Any:
        try:
            return self.last()
        except NoElementError:
            return default

    @overload
    def cast(self, type: Type[Tuple[K2, V2]]) -> PairQuery[K2, V2]:
        pass

    @overload
    def cast(self, type: Type[R]) -> Query[R]:
        pass

    @overload
    def cast(self, type: Callable[[Tuple[K, V]], Tuple[K2, V2]]) -> PairQuery[K2, V2]:
        pass

    @overload
    def cast(self, type: Callable[[Tuple[K, V]], R]) -> Query[R]:
        pass

    def cast(self, type: Type[R]) -> Query[R]:
        return self

    @lazy_iterate
    def enumerate(self, start: int = 0, step: int = 1) -> PairQuery[int, Tuple[K, V]]:
        yield from (x * step for x in enumerate(self, start))

    @overload
    def map(self, type: Callable[[Tuple[K, V]], Tuple[K2, V2]]) -> PairQuery[K2, V2]:
        pass

    @overload
    def map(self, type: Callable[[Tuple[K, V]], R]) -> Query[R]:
        pass

    @lazy_iterate
    def map(self, selector: Callable[[Tuple[K, V]], R]) -> Query[R]:
        if selector is str:
            selector = lambda x: "" if x is None else str(x)
        yield from map(selector, self)

    @overload
    def select(self, item: Literal[0]) -> "Query[K]":
        ...

    @overload
    def select(self, item: Literal[1]) -> "Query[V]":
        ...

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
    def unpack(self, selector: Callable[..., R]) -> Query[R]:
        for elm in self:
            yield selector(*elm)  # type: ignore

    @lazy_iterate
    def unpack_kw(self, selector: Callable[..., R]) -> Query[R]:
        for elm in self:
            yield selector(**elm)  # type: ignore

    @lazy_iterate
    def group_by(
        self, selector: Callable[[Tuple[K, V]], Tuple[K2, V2]]
    ) -> PairQuery[K2, List[V2]]:
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
            lambda left, right: (left.name, right.age),
        )

        table(User).join(Item, on=User.id == Item.id).select(User.id, Item.id)

        pass

    @lazy_iterate
    def distinct(
        self, selector: Callable[[Tuple[K, V]], Any], msg: str = ...
    ) -> PairQuery[K, V]:
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
    def filter(self, predicate: Callable[[Tuple[K, V]], bool]) -> PairQuery[K, V]:
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

    @lazy_iterate
    def must(
        self, predicate: Callable[[Tuple[K, V]], bool], msg: str = ...
    ) -> PairQuery[K, V]:
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
    def skip(self, count: int) -> PairQuery[K, V]:
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
    def take(self, count: int) -> PairQuery[K, V]:
        current = 0

        try:
            while current < count:
                yield next(self)
                current += 1
        except StopIteration:
            return

    @lazy_iterate
    def range(self, start: int = ..., stop: int = ...) -> PairQuery[K, V]:
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
    def page(self, page: int = ..., size: int = ...) -> PairQuery[K, V]:
        start, stop = page_calc(page, size)
        yield from self.range(start, stop)

    @lazy_reference
    def reverse(self) -> PairQuery[K, V]:
        yield actions.reverse(self)

    @lazy_iterate
    def order(self, selector, desc: bool = False) -> PairQuery[K, V]:
        yield from sorted(self, key=selector, reverse=desc)

    def order_by_items(self, *items: Any, desc: bool = False) -> PairQuery[K, V]:
        selector = itemgetter(*items)
        return self.order(selector, desc)

    def order_by_attrs(
        self: Iterable[T], *attrs: str, desc: bool = False
    ) -> PairQuery[K, V]:
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


class IndexQuery(Generic[K, V]):
    @lazy_reference
    def get_many(self, *keys: K) -> IndexQuery[K, V]:
        undefined = object()
        for id in keys:
            obj = self.get(id, undefined)
            if not obj is undefined:
                yield id, obj

    @overload
    def get(self, key: K) -> V:
        ...

    @overload
    def get(self, key: K, default: R = ...) -> Union[V, R]:
        ...

    def get(self, key: K, default: R = undefined) -> Any:
        try:
            return self[key]  # type: ignore
        except (KeyError, IndexError):
            if default is not undefined:
                return default
            else:
                raise

    def get_or_default(self, key, default=None):
        return self.get(key, default)  # type: ignore

    def get_or_none(self, key):
        return self.get(key, None)  # type: ignore

    def to_list(self) -> ListEx[Tuple[K, V]]:
        return ListEx(piter(self))

    def to_dict(self) -> DictEx[K, V]:
        return DictEx(piter(self))


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


def query(source: T) -> T:
    if hasattr(source, "__piter__"):
        return source
    elif isinstance(source, dict):
        return DictEx(source)
    elif isinstance(source, list):
        return ListEx(source)
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
