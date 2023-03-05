import asyncio
from concurrent.futures import Future as _Future
from typing import Iterable, Mapping, Protocol


class PArguments(Protocol):
    args: Iterable
    kwargs: Mapping


class Futurable:
    def __future__(self) -> _Future:
        ...


class AsyncFuturable:
    def __afuture__(self) -> asyncio.Future:
        ...
