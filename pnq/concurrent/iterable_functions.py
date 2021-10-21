from typing import AsyncIterable, Callable, Iterable, TypeVar, Union, overload

from . import iterables

T = TypeVar("T")


@overload
def LazyGeneratorFunction(
    func: Union[Callable[..., Iterable[T]], Callable[..., AsyncIterable[T]]]
) -> iterables.LazyGeneratorFunction[T]:
    ...


@overload
def LazyGeneratorFunction(func) -> iterables.LazyGeneratorFunction:
    ...


def LazyGeneratorFunction(func):
    return iterables.LazyGeneratorFunction(func)


@overload
def LazyListable(func: AsyncIterable[T]) -> iterables.AsyncListable[T]:  # type: ignore
    ...


@overload
def LazyListable(func: Iterable[T]) -> iterables.Listable[T]:  # type: ignore
    ...


@overload
def LazyListable(func) -> iterables.LazyListable:
    ...


def LazyListable(func):
    return iterables.LazyListable(func)
