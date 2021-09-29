from typing import (
    Any,
    AsyncIterator,
    Dict,
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


def it(type: Type[T]) -> Iterator[T]:
    ...


async def ait(type: Type[T]) -> AsyncIterator[T]:
    ...


def query_empty() -> Iterator[Query]:
    yield 1  # type: ignore
    yield pnq.query([])
    yield pnq.query({})
    yield pnq.query(set())
    yield pnq.query(tuple())
    yield pnq.query(it(int))
    yield pnq.query(ait(int))


def query_int() -> Iterator[Query[int]]:
    yield 1  # type: ignore
    yield pnq.query([1])
    yield pnq.query({"a": 1})  # type: ignore
    yield pnq.query(set([1]))
    yield pnq.query(tuple([1]))
    yield pnq.query(it(int))
    yield pnq.query(ait(int))


def query_tuple() -> Iterator[Query[Tuple[str, int]]]:
    yield 1  # type: ignore
    yield pnq.query([1])
    yield pnq.query({"a": 1})
    yield pnq.query(set([1]))
    yield pnq.query(tuple([1]))
    yield pnq.query(it(int))
    yield pnq.query(ait(int))


def query_pair() -> Iterator[PairQuery[str, int]]:
    yield 1  # type: ignore
    yield pnq.query([1])  # type: ignore
    yield pnq.query({"a": 1})
    yield pnq.query(set([1]))  # type: ignore
    yield pnq.query(tuple([1]))  # type: ignore
    yield pnq.query(it(int))  # type: ignore
    yield pnq.query(ait(int))  # type: ignore


def test_to() -> Iterator[List]:
    yield 1  # type: ignore
    yield pnq.query([1]).to(list)
    yield pnq.query({"a": 1}).to(list)
    yield pnq.query(set([1])).to(list)
    yield pnq.query(tuple([1])).to(list)
    yield pnq.query(it(int)).to(list)
    yield pnq.query(ait(int)).to(list)
