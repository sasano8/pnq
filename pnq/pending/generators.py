from typing import TYPE_CHECKING, Iterable, Tuple, TypeVar, overload

from .commons import name_as
from .core import LazyGenerator
from .queries import DictEx, ListEx, query

if TYPE_CHECKING:
    from _typeshed import SupportsKeysAndGetItem

T = TypeVar("T")
K = TypeVar("K")
V = TypeVar("V")


@name_as("list")
def __list(iterable: Iterable[T]) -> ListEx[T]:
    return ListEx(iterable)


@overload
def __dict(mapping: "SupportsKeysAndGetItem[K, V]") -> DictEx[K, V]:
    return DictEx(mapping)


@overload
def __dict(iterable: Iterable[Tuple[K, V]]) -> DictEx[K, V]:
    return DictEx(iterable)


@overload
def __dict(iterable) -> DictEx:
    return DictEx(iterable)


@name_as("dict")
def __dict(iterable) -> DictEx:
    return DictEx(iterable)


def infinite(func, *args, **kwargs):
    def infinite(*args, **kwargs):
        while True:
            yield func(*args, **kwargs)

    return query(LazyGenerator(infinite, *args, **kwargs))


def count(start=0, step=1):
    from itertools import count

    return query(LazyGenerator(count, start, step))


def cycle(iterable):
    from itertools import cycle

    return query(LazyGenerator(cycle, iterable))
