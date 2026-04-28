import asyncio
import concurrent.futures
from typing import Protocol

from pnq.protocols import IterType, PAsyncResult, PResult, WrappedQuery

__all__ = ["PExecutor", "PResult", "PAsyncResult", "IterType", "PQuery"]


class PExecutor(Protocol):
    def submit(self, func, *args, **kwargs) -> concurrent.futures.Future: ...

    def asubmit(self, func, *args, **kwargs) -> asyncio.Future: ...

    @property
    def is_cpubound(self) -> bool: ...

    @property
    def max_workers(self) -> int: ...

    @property
    def is_async_only(self) -> bool: ...


# pnq.protocols.WrappedQuery は Iterable / AsyncIterable / PResult /
# PAsyncResult / iter_type を備えており、従来の PQuery と同じ役割を果たす。
PQuery = WrappedQuery
