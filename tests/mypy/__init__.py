from typing import (
    Any,
    AsyncIterator,
    Dict,
    Generic,
    Iterable,
    Iterator,
    List,
    Tuple,
    Type,
    TypeVar,
    overload,
)

import pnq
from pnq.queries import PairQuery, Query, QueryDict, QuerySeq, QuerySet

T = TypeVar("T")


class IT(Generic[T]):
    def __iter__(self) -> Iterator[T]:
        ...


class AIT(Generic[T]):
    def __aiter__(self) -> AsyncIterator[T]:
        ...


def query_empty() -> Iterator[Query]:
    yield 1  # type: ignore
    yield pnq.query([])
    yield pnq.query({})
    yield pnq.query(set())
    yield pnq.query(tuple())
    yield pnq.query(IT[int]())
    yield pnq.query(AIT[int]())


def query_int() -> Iterator[Query[int]]:
    yield 1  # type: ignore
    yield pnq.query([1])
    yield pnq.query({"a": 1})  # type: ignore
    yield pnq.query(set([1]))
    yield pnq.query(tuple([1]))
    yield pnq.query(IT[int]())
    yield pnq.query(AIT[int]())


def query_tuple() -> Iterator[Query[Tuple[str, int]]]:
    yield 1  # type: ignore
    yield pnq.query([("a", 1)])
    yield pnq.query({"a": 1})
    yield pnq.query(set([("a", 1)]))
    yield pnq.query(tuple([("a", 1)]))
    yield pnq.query(IT[Tuple[str, int]]())
    yield pnq.query(AIT[Tuple[str, int]]())


def query_pair() -> Iterator[PairQuery[str, int]]:
    yield 1  # type: ignore
    yield pnq.query([("a", 1)])
    yield pnq.query({"a": 1})
    yield pnq.query(set([("a", 1)]))
    yield pnq.query(tuple([("a", 1)]))
    yield pnq.query(IT[Tuple[str, int]]())
    yield pnq.query(AIT[Tuple[str, int]]())


def test_to() -> Iterator[List[Any]]:
    yield 1  # type: ignore
    yield pnq.query([1]).to(list)
    yield pnq.query({"a": 1}).to(list)
    yield pnq.query(set([1])).to(list)
    yield pnq.query(tuple([1])).to(list)
    yield pnq.query(IT[int]()).to(list)
    yield pnq.query(AIT[int]()).to(list)
