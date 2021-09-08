from asyncio import sleep as asleep
from functools import wraps
from time import sleep
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Generic,
    Iterable,
    Iterator,
    List,
    Mapping,
    Reversible,
    Tuple,
    Type,
    TypeVar,
    Union,
    overload,
)

T = TypeVar("T")
K = TypeVar("K")
V = TypeVar("V")
R = TypeVar("R")
R2 = TypeVar("R2")
T_OTHER = TypeVar("T_OTHER")
F = TypeVar("F", bound=Callable)
undefined = object()


__all__ = ["plen", "piter", "pfilter", "pmap"]


def plen(target) -> int:
    if hasattr(target, "__plen__"):
        return target.__plen__()
    else:
        return len(list(target))


@overload
def piter(target: Mapping[K, V]) -> Iterator[Tuple[K, V]]:  # type: ignore
    ...


@overload
def piter(target: Iterable[T]) -> Iterator[T]:
    ...


def piter(target):
    if hasattr(target, "__piter__"):
        return target.__piter__()
    elif isinstance(target, Mapping):
        return iter(target.items())
    else:
        return iter(target)


@overload
def pfilter(  # type: ignore
    target: Mapping[K, V], predicate: Callable[[Tuple[K, V]], bool]
) -> Iterator[Tuple[K, V]]:
    ...


@overload
def pfilter(target: Iterable[T], predicate: Callable[[T], bool]) -> Iterator[T]:
    ...


def pfilter(target, predicate):
    return filter(predicate, piter(target))


@overload
def pmap(  #  type: ignore
    target: Mapping[K, V], selector: None = None
) -> Iterator[Tuple[K, V]]:
    ...


@overload
def pmap(target: Mapping[K, V], selector: Callable[[Tuple[K, V]], R]) -> Iterator[R]:
    ...


@overload
def pmap(target: Iterable[T], selector: None = None) -> Iterator[T]:
    ...


@overload
def pmap(target: Iterable[T], selector: Callable[[T], R]) -> Iterator[R]:
    ...


def pmap(target, selector=None):
    if selector is None:
        return piter(target)
    else:
        return map(selector, piter(target))


def penumerate(self):
    return enumerate(piter(self))


pfilter.__name__ = "filter"
pmap.__name__ = "map"
penumerate.__name__ = "enumerate"

# a = piter([1])
# b = piter({1: "a"})

# c = pfilter([1], lambda x: x % 2 == 0)
# d = pfilter({1: "a"}, lambda x: x % 2 == 0)

# e = pmap([1])
# f = pmap([1], lambda x: "test")
# g = pmap({1: "a"})
# h = pmap({1: "a"}, lambda x: 0.1)
