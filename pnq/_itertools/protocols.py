import asyncio
import concurrent.futures

from typing_extensions import Protocol


class PExecutor(Protocol):
    def submit(self, func, *args, **kwargs) -> concurrent.futures.Future:
        ...

    def asubmit(self, func, *args, **kwargs) -> asyncio.Future:
        ...

    @property
    def is_cpubound(self) -> bool:
        ...