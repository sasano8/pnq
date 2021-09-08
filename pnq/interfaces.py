import inspect
import statistics
from asyncio import sleep as asleep
from collections import defaultdict
from functools import wraps
from itertools import zip_longest
from operator import is_not
from time import sleep
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Generic,
    ItemsView,
    Iterable,
    Iterator,
    KeysView,
    List,
    Mapping,
    MutableMapping,
    Protocol,
    Reversible,
    Sequence,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
    ValuesView,
    cast,
    no_type_check,
    overload,
)

from . import reflectors
from .exceptions import NoElementError, NotOneError
from .getter import attrgetter, itemgetter

T = TypeVar("T")
K = TypeVar("K")
V = TypeVar("V")
K2 = TypeVar("K2")
V2 = TypeVar("V2")

R = TypeVar("R")
R2 = TypeVar("R2")
Q = TypeVar("Q", bound=Iterable)
T_OTHER = TypeVar("T_OTHER")
F = TypeVar("F", bound=Callable)


from .buildins import penumerate, pfilter, piter, plen, pmap
from .protocol import KeyValueItems, undefined


def query(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        return SeqQuery(func, *args, **kwargs)

    return wrapper


def query_for_instance(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        return InstanceQuery(func, *args, **kwargs)

    return wrapper


class KeyValueIterable(Generic[K, V]):
    if TYPE_CHECKING:

        def items(self) -> ItemsView[K, V]:
            raise NotImplementedError()


class Action(Generic[T]):
    if TYPE_CHECKING:

        def __piter__(self) -> Iterator[T]:
            raise NotImplementedError()

        def __plen__(self) -> Iterator[T]:
            raise NotImplementedError()

        def __iter__(self) -> Iterator[T]:
            raise NotImplementedError()

        def __reversed__(self: Iterable[T]) -> Iterator[T]:
            raise NotImplementedError()

        # def __getitem__(self, key) -> T:
        #     raise NotImplementedError()

        def __contains__(self, key) -> bool:
            raise NotImplementedError()

    def len(self: Iterable[T]) -> int:
        return plen(self)

    def exists(self: Iterable[T]) -> bool:
        return bool(Action.len(self))

    @overload
    def one(self: Iterable[T]) -> T:
        ...

    @overload
    def one(self: Iterable[T], selector: Callable[[T], R] = lambda x: x) -> R:  # type: ignore
        ...

    def one(self: Iterable[T], selector: Callable[[T], R] = lambda x: x) -> R:  # type: ignore
        it = piter(self)
        try:
            result = next(it)
        except StopIteration:
            raise NoElementError()

        try:
            next(it)
        except StopIteration:
            raise NotOneError()

        return selector(result)

    @overload
    def first(self: Iterable[T]) -> T:
        ...

    @overload
    def first(self: Iterable[T], selector: Callable[[T], R] = lambda x: x) -> R:  # type: ignore
        ...

    def first(self: Iterable[T], selector=lambda x: x):
        if isinstance(self, Sequence):
            obj = self[0]
        else:
            try:
                it = piter(self)
                obj = next(it)
            except StopIteration:
                raise NoElementError()

        return selector(obj)

    @overload
    def last(self: Iterable[T]) -> T:
        ...

    @overload
    def last(self: Iterable[T], selector: Callable[[T], R] = lambda x: x) -> R:  # type: ignore
        ...

    def last(self: Iterable[T], selector: Callable[[T], R] = lambda x: x) -> R:  # type: ignore
        undefined = object()
        last: R = undefined  # type: ignore
        for elm in piter(self):
            last = elm  # type: ignore

        if last is undefined:
            raise NoElementError()

        return selector(last)  # type: ignore

    @overload
    def one_or_default(
        self: Iterable[T],
    ) -> Union[T, None]:
        ...

    @overload
    def one_or_default(
        self: Iterable[T],
        default: R,
    ) -> Union[T, R]:
        ...

    def one_or_default(
        self: Iterable[T],
        default: R = None,
    ):
        try:
            return Action.one(self)
        except (NoElementError, NotOneError):
            return default

    @overload
    def first_or_default(
        self: Iterable[T],
    ) -> Union[T, None]:
        ...

    @overload
    def first_or_default(
        self: Iterable[T],
        default: R,
    ) -> Union[T, R]:
        ...

    def first_or_default(
        self: Iterable[T],
        default: R = None,
    ):
        try:
            return Action.first(self)
        except NoElementError:
            return default

    @overload
    def last_or_default(
        self: Iterable[T],
    ) -> Union[T, None]:
        ...

    @overload
    def last_or_default(
        self: Iterable[T],
        default: R,
    ) -> Union[T, R]:
        ...

    def last_or_default(
        self: Iterable[T],
        default: R = None,
    ):
        try:
            return Action.last(self)
        except NoElementError:
            return default

    def to_index(self, duplicate: bool = True) -> "DictEx":
        return DictEx(piter(self))  # type: ignore

    def to_list(self) -> "ListEx":
        return ListEx(piter(self))


class Indexer(Generic[K, V]):
    @query_for_instance
    def get_many(self, *keys: K) -> "PairQuery[K, V]":  # type: ignore
        undefined = object()
        for id in keys:
            obj = Indexer.get(self, id, undefined)
            if not obj is undefined:
                yield id, obj

    @overload
    def get(self, key: K) -> V:
        pass

    @overload
    def get(self, key: K, default: R = undefined) -> Union[V, R]:
        pass

    def get(self, key: K, default: R = undefined):
        try:
            return self[key]  # type: ignore
        except (KeyError, IndexError):
            if default is not undefined:
                return default
            else:
                raise


class SeqAction(Action[T]):
    @no_type_check
    def get(
        self: Iterable[T],
        key: int,
        default: R = undefined,
    ) -> R:
        try:
            result = SeqFilter.range(self, key, key + 1).one()
        except (NoElementError, NotOneError):
            if default is not undefined:
                return default
            raise IndexError()

        return result

    def all(self: Iterable[T], selector: Callable[[T], Any] = lambda x: x) -> bool:
        return all(pmap(self, selector))

    def any(self: Iterable[T], selector: Callable[[T], Any] = lambda x: x) -> bool:
        return any(pmap(self, selector))

    @overload
    def min(self: Iterable[T]) -> T:
        ...

    @overload
    def min(self: Iterable[T], selector: Callable[[T], R] = lambda x: x) -> R:  # type: ignore
        ...

    def min(self: Iterable[T], selector: Callable[[T], R] = lambda x: x) -> R:  # type: ignore
        return min(pmap(self, selector))  # type: ignore

    @overload
    def max(self: Iterable[T]) -> T:
        ...

    @overload
    def max(self: Iterable[T], selector: Callable[[T], R] = lambda x: x) -> R:  # type: ignore
        ...

    def max(self: Iterable[T], selector: Callable[[T], R] = lambda x: x) -> R:  # type: ignore
        return max(pmap(self, selector))  # type: ignore

    @overload
    def sum(self: Iterable[T]) -> T:
        ...

    @overload
    def sum(self: Iterable[T], selector: Callable[[T], R] = lambda x: x) -> R:  # type: ignore
        ...

    def sum(self: Iterable[T], selector: Callable[[T], R] = lambda x: x) -> R:  # type: ignore
        return sum(pmap(self, selector))  # type: ignore

    @overload
    def average(self: Iterable[T]) -> T:
        ...

    @overload
    def average(self: Iterable[T], selector: Callable[[T], R] = lambda x: x) -> R:  # type: ignore
        ...

    def average(self: Iterable[T], selector: Callable[[T], R] = lambda x: x) -> R:  # type: ignore
        return statistics.mean(pmap(self, selector))  # type: ignore

    def reduce(
        self: Iterable[T],
        accumulator: Callable[[T, T], T],
        seed: T = undefined,
    ):
        from functools import reduce

        if seed is undefined:
            return reduce(accumulator, piter(self))
        else:
            return reduce(accumulator, piter(self), seed)

    def dispatch(
        self: Iterable[T],
        func: Callable,
        selector: Callable[[T], Any],
        on_error: Callable = lambda func, v, err: False,
    ) -> Tuple[int, int]:
        success = 0
        errors = 0

        for elm in piter(self):
            value = selector(elm)
            try:
                func(value)
                success += 1
            except Exception as e:
                ignore_err = on_error(func, value, e)
                if not ignore_err:
                    raise e
                errors += 1

        return success, errors

    if TYPE_CHECKING:

        def to_list(self: Iterable[T]) -> "ListEx[T]":
            ...


class SeqTransform(Action[T]):
    def cast(self, type: Type[R]) -> "SeqQuery[R]":
        return self  # type: ignore

    @query
    def enumerate(self: Iterable[T], start: int = 0) -> "PairQuery[int, T]":
        return enumerate(self, start)  # type: ignore

    @query
    def map(self: Iterable[T], selector: Callable[[T], R]) -> "SeqQuery[R]":
        return map(selector, self)  # type: ignore

    @query
    def unpack(self: Iterable[T], selector: Callable[..., R]) -> "SeqQuery[R]":  # type: ignore
        for elm in piter(self):
            yield selector(*elm)  # type: ignore

    @query
    def unpack_kw(self: Iterable[T], selector: Callable[..., R]) -> "SeqQuery[R]":  # type: ignore
        for elm in piter(self):
            yield selector(**elm)  # type: ignore

    @query
    def group_by(  # type: ignore
        self: Iterable[T], selector: Callable[[T], Tuple[K, V]]
    ) -> "PairQuery[K, List[V]]":
        results: Dict[K, List[V]] = defaultdict(list)
        for elm in piter(self):
            k, v = selector(elm)
            results[k].append(v)

        for k, v in results.items():  # type: ignore
            yield k, v

    def get_many(self: Iterable[T], keys: Iterable[int]) -> "SeqQuery[T]":
        pass

    @query
    def pairs(  # type: ignore
        self: Iterable[T], selector: Callable[[T], Tuple[K, V]] = lambda x: x
    ) -> "PairQuery[K, V]":
        for elm in self:
            yield selector(elm)


class SeqFilter(Action[T]):
    @query
    def filter(self: Iterable[T], predicate: Callable[[T], bool]) -> "SeqQuery[T]":
        return filter(predicate, self)  # type: ignore

    @query
    def must(  # type: ignore
        self: Iterable[T], predicate: Callable[[T], bool], msg: str = ""
    ) -> "SeqQuery[T]":
        """要素の検証に失敗した時例外を発生させる。"""
        for elm in self:
            if not predicate(elm):
                raise ValueError(f"{msg} {elm}")
            yield elm

    @query
    def unique(  # type: ignore
        self: Iterable[T], selector: Callable[[T], Any], msg: str = ""
    ) -> "SeqQuery[T]":
        duplicate = set()
        for elm in self:
            value = selector(elm)
            if not value in duplicate:
                duplicate.add(value)
                yield elm

    @query
    def skip(self: Iterable[T], count: int) -> "SeqQuery[T]":  # type: ignore
        it = iter(self)
        current = 0

        try:
            while current < count:
                next(it)
                current += 1
        except StopIteration:
            return

        for elm in it:
            yield elm

    @query
    def take(self: Iterable[T], count: int) -> "SeqQuery[T]":  # type: ignore
        it = iter(self)
        current = 0

        try:
            while current < count:
                yield next(it)
                current += 1
        except StopIteration:
            return

    @query
    def range(self: Iterable[T], start: int = 0, stop: int = None) -> "SeqQuery[T]":  # type: ignore
        it = iter(self)
        if start < 0:
            start = 0

        if stop is None:
            stop = float("inf")  # type: ignore
        elif stop < 0:
            stop = 0
        else:
            pass

        current = 0

        try:
            while current < start:
                next(it)
                current += 1
        except StopIteration:
            return

        try:
            while current < stop:  # type: ignore
                yield next(it)
                current += 1
        except StopIteration:
            return

    @staticmethod
    def _page_calc(page: int, size: int):
        if size < 0:
            raise ValueError("size must be >= 0")
        start = (page - 1) * size
        stop = start + size
        return start, stop

    def page(self: Iterable[T], page: int = 1, size: int = 0) -> "SeqQuery[T]":
        start, stop = SeqFilter._page_calc(page, size)
        return SeqFilter.range(self, start, stop)


class SeqSorter(Action[T]):
    @query
    def reverse(self) -> "SeqQuery[T]":  # type: ignore
        if hasattr(self, "__reversed__"):
            yield from self.__reversed__()
        else:
            yield from list(self).__reversed__()


class KeyValueSorter(KeyValueIterable[K, V]):
    @query
    def reverse(self: KeyValueIterable[K, V]) -> "PairQuery[K, V]":  # type: ignore
        if hasattr(self, "__reversed__"):
            yield from ((k, self[k]) for k in self.__reversed__())  # type: ignore
        else:
            yield from SeqSorter.reverse(self.items())


class KeyValueAction(KeyValueIterable[K, V], Generic[K, V]):
    def all(
        self: KeyValueIterable[K, V],
        selector: Callable[[Tuple[K, V]], Any] = lambda x: x[1],
    ) -> bool:
        return SeqAction.all(self, selector)  # type: ignore

    def any(
        self: KeyValueIterable[K, V],
        selector: Callable[[Tuple[K, V]], Any] = lambda x: x[1],
    ) -> bool:
        return SeqAction.any(self, selector)  # type: ignore

    @overload
    def min(
        self: KeyValueIterable[K, V],
    ) -> V:
        ...

    @overload
    def min(
        self: KeyValueIterable[K, V],
        selector: Callable[[Tuple[K, V]], R] = lambda x: x[1],  # type: ignore
    ) -> R:
        ...

    def min(
        self: KeyValueIterable[K, V],
        selector: Callable[[Tuple[K, V]], R] = lambda x: x[1],  # type: ignore
    ):
        return SeqAction.min(self, selector)  # type: ignore

    @overload
    def max(
        self: KeyValueIterable[K, V],
    ) -> V:
        ...

    @overload
    def max(
        self: KeyValueIterable[K, V],
        selector: Callable[[Tuple[K, V]], R] = lambda x: x[1],  # type: ignore
    ) -> R:
        ...

    def max(
        self: KeyValueIterable[K, V],
        selector: Callable[[Tuple[K, V]], R] = lambda x: x[1],  # type: ignore
    ):
        return SeqAction.max(self, selector)  # type: ignore

    @overload
    def sum(
        self: KeyValueIterable[K, V],
    ) -> V:
        ...

    @overload
    def sum(
        self: KeyValueIterable[K, V],
        selector: Callable[[Tuple[K, V]], R] = lambda x: x[1],  # type: ignore
    ) -> R:
        ...

    def sum(
        self: KeyValueIterable[K, V],
        selector: Callable[[Tuple[K, V]], R] = lambda x: x[1],  # type: ignore
    ):
        return SeqAction.sum(self, selector)  # type: ignore

    @overload
    def average(
        self: KeyValueIterable[K, V],
    ) -> V:
        ...

    @overload
    def average(
        self: KeyValueIterable[K, V],
        selector: Callable[[Tuple[K, V]], R] = lambda x: x[1],  # type: ignore
    ) -> R:
        ...

    def average(
        self: KeyValueIterable[K, V],
        selector: Callable[[Tuple[K, V]], R] = lambda x: x[1],  # type: ignore
    ):
        return SeqAction.average(self, selector)  # type: ignore

    def accumulate(self):
        """累積の経過を返すと思う"""
        pass

    def reduce(
        self: KeyValueIterable[K, V],
        accumulator: Callable[[Tuple[K, V], R], R] = lambda x, y: x[1] + y,  # type: ignore
        seed: R = undefined,
    ):
        return SeqAction.reduce(self, accumulator, seed)  # type: ignore

    def dispatch(
        self: KeyValueIterable[K, V],
        func: Callable,
        selector: Callable[[Tuple[K, V]], Any] = lambda x: x[1],
        on_error: Callable = lambda func, value, err: False,
    ) -> Tuple[int, int]:
        return SeqAction.dispatch(self, func, selector, on_error)  # type: ignore


class KeyValueTransform(KeyValueIterable[K, V], Generic[K, V]):
    @query
    def map(
        self: KeyValueIterable[K, V], selector: Callable[[Tuple[K, V]], R]
    ) -> "SeqQuery[R]":
        return map(selector, self)  # type: ignore


class DictFilter(KeyValueIterable[K, V], Generic[K, V]):
    def reverse(self: KeyValueIterable[K, V]):
        pass


class QueryBase:
    # def __init__(self, func, source: Iterable[T], /, *args, **kwargs) -> None:
    def __init__(self, func, source: Iterable[T], *args, **kwargs) -> None:
        self.func = func
        self.source = source
        self.args = args
        self.kwargs = kwargs

    @no_type_check
    def __iter__(self) -> Iterator[T]:
        return self.__piter__()

    def __str__(self) -> str:
        kwargs = [f"{k}={v}" for k, v in self.kwargs.items()]
        params = list(self.args) + kwargs
        params_str = ", ".join(str(self.__inspect_arg(x)) for x in params)
        return f"{self.func.__name__}({params_str})"

    def debug(self):
        info = self.get_upstream(self)
        str_info = [str(x) for x in info]
        info = ".".join(str_info)
        print(info)
        return info

    @classmethod
    def __get_upstream(cls, target):
        if hasattr(target, "source"):
            source = target.source
            return source
        else:
            return None

    @classmethod
    def _get_upstream(cls, target):
        if not target:
            raise TypeError()

        results = [target]

        while target:
            target = cls.__get_upstream(target)
            if target:
                results.append(target)

        return results

    @classmethod
    def get_upstream(cls, target):
        pipe = cls._get_upstream(target)
        pipe.reverse()
        info = [x for x in pipe]
        return info

    @staticmethod
    def __inspect_arg(arg):
        if inspect.isfunction(arg):
            if arg.__name__ == "<lambda>":
                # return inspect.getsourcelines(arg)[0][0]
                sig = inspect.signature(arg)
                return str(sig) + " => ..."
            else:
                return arg.__name__

        elif isinstance(arg, type):
            return arg.__name__
        else:
            return arg


class SeqQuery(
    QueryBase, Iterable[T], SeqAction[T], SeqTransform[T], SeqFilter[T], SeqSorter[T]
):
    def __piter__(self):
        return self.func(piter(self.source), *self.args, **self.kwargs)


class InstanceQuery(SeqQuery[T]):
    def __piter__(self):
        return self.func(self.source, *self.args, **self.kwargs)


class PairQuery(
    QueryBase,
    # KeyValueIterable[K, V],
    # Iterable[Tuple[K, V]],
    # SeqAction[Tuple[K, V]],
    # SeqTransform[Tuple[K, V]],
    # SeqFilter[Tuple[K, V]],
    KeyValueSorter[K, V],
):
    if TYPE_CHECKING:

        def items(self) -> ItemsView[K, V]:
            raise NotImplementedError()

    def to_index(
        self: KeyValueIterable[K, V], duplicate: bool = False
    ) -> "DictEx[K, V]":
        raise NotImplementedError()

    def filter(
        self: KeyValueIterable[K, V], predicate: Callable[[T], bool]
    ) -> "PairQuery[K, V]":
        raise NotImplementedError()

    def must(
        self: KeyValueIterable[K, V], predicate: Callable[[T], bool], msg: str = ""
    ) -> "PairQuery[K, V]":
        """要素の検証に失敗した時例外を発生させる。"""
        raise NotImplementedError()

    def unique(
        self: KeyValueIterable[K, V], selector: Callable[[T], Any], msg: str = ""
    ) -> "PairQuery[K, V]":
        raise NotImplementedError()

    def skip(self: KeyValueIterable[K, V], count: int) -> "PairQuery[K, V]":
        raise NotImplementedError()

    def take(self: KeyValueIterable[K, V], count: int) -> "PairQuery[K, V]":
        raise NotImplementedError()

    def range(
        self: KeyValueIterable[K, V], start: int = 0, stop: int = None
    ) -> "PairQuery[K, V]":
        raise NotImplementedError()

    def page(  # type: ignore
        self: KeyValueIterable[K, V], page: int = 1, size: int = 0
    ) -> "PairQuery[K, V]":
        raise NotImplementedError()

    @query
    def enumerate(
        self: KeyValueIterable[K, V], start: int = 0
    ) -> "PairQuery[int, Tuple[K, V]]":
        raise NotImplementedError()

    def map(self: KeyValueIterable[K, V], selector: Callable[[T], R]) -> "SeqQuery[R]":
        raise NotImplementedError()

    def pairs(
        self: KeyValueIterable[K, V], selector: Callable[[Tuple[K, V]], Tuple[K2, V2]]
    ) -> "PairQuery[K2, V2]":
        raise NotImplementedError()

    def unpack(
        self: KeyValueIterable[K, V], selector: Callable[..., R]
    ) -> "SeqQuery[R]":
        raise NotImplementedError()

    def unpack_kw(
        self: KeyValueIterable[K, V], selector: Callable[..., R]
    ) -> "SeqQuery[R]":
        raise NotImplementedError()

    def group_by(
        self: KeyValueIterable[K, V], selector: Callable[[Tuple[K, V]], Tuple[K2, V2]]
    ) -> "PairQuery[K2, List[V2]]":
        raise NotImplementedError()


# 左が一番優先される
class ListEx(
    Indexer[int, T], SeqAction[T], SeqTransform[T], SeqFilter[T], SeqSorter[T], List[T]
):
    def __iter__(self) -> Iterator[T]:
        return super().__iter__()

    def __piter__(self) -> Iterator[T]:
        return super().__iter__()


# 左が上書きされる
class DictEx(Indexer[K, V], KeyValueTransform[K, V], Dict[K, V]):
    def __piter__(self) -> ItemsView[K, V]:
        return self.items()

    @query
    def reverse(self) -> "PairQuery[K, V]":  # type: ignore
        keys = reversed(self)
        yield from ((k, self[k]) for k in keys)

    def to_index(self, duplicate: bool = True) -> "DictEx[K, V]":
        return DictEx(self)


class SetEx(Indexer[T, T], Set[T], SeqQuery[T]):
    pass
