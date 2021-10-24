import asyncio
import concurrent.futures
from enum import Flag
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterable,
    Awaitable,
    Generator,
    Generic,
    Iterable,
    List,
    TypeVar,
)

from typing_extensions import Protocol

T = TypeVar("T")


class PExecutor(Protocol):
    def submit(self, func, *args, **kwargs) -> concurrent.futures.Future:
        ...

    def asubmit(self, func, *args, **kwargs) -> asyncio.Future:
        ...

    @property
    def is_cpubound(self) -> bool:
        ...

    @property
    def max_workers(self) -> int:
        ...

    @property
    def is_async_only(self) -> bool:
        ...


class PResult(Protocol[T]):
    def result(self, timeout=None) -> List[T]:
        ...


class PAsyncResult(Awaitable[List[T]], Generic[T]):
    def __await__(self) -> Generator[Any, Any, List[T]]:
        ...


class IterType(Flag):
    IMPOSSIBLE = 0
    NORMAL = 1
    ASYNC = 2
    BOTH = NORMAL | ASYNC


class PQuery(Iterable[T], AsyncIterable[T], PResult[T], PAsyncResult[T], Generic[T]):
    iter_type: IterType
