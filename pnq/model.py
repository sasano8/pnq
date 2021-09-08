from typing import (
    TYPE_CHECKING,
    Dict,
    Iterable,
    List,
    Mapping,
    Tuple,
    TypeVar,
    Union,
    cast,
)

from .interfaces import (
    DictAction,
    DictTransform,
    SeqAction,
    SeqFilter,
    SeqSorter,
    SeqTransform,
)
from .protocol import KeyValueItems

T = TypeVar("T")
K = TypeVar("K")
V = TypeVar("V")


class PnqList(List[T], SeqAction[T], SeqTransform[T], SeqFilter[T], SeqSorter[T]):
    def add(self, value):
        self.append(value)


class PnqDict(Dict[K, V], DictAction[K, V], DictTransform[K, V]):
    pass


def func(mapping: KeyValueItems[K, V]) -> Iterable[Tuple[K, V]]:
    return mapping  # type: ignore


a = func({1: ""})


class PnqTuple(tuple):
    pass


class PnqSet(set):
    def append(self, value):
        self.add(value)


class PnqFronzenset(frozenset):
    pass
