import asyncio
import concurrent
from typing import Dict, Iterable

from typing_extensions import Protocol, runtime_checkable  # from python3.8

"""
concurrent.futures.ProcessPoolExecutorはmultiprocessing.Poolのラッパーです。
https://stackoverflow.com/questions/38311431/concurrent-futures-processpoolexecutor-vs-multiprocessing-pool-pool
"""


@runtime_checkable
class Executor(Protocol):
    def submit(self, *args, **kwargs) -> concurrent.futures.Future:
        ...


class PExecutor(Protocol):
    def submit(self, func, *args, **kwargs) -> concurrent.futures.Future:
        ...

    def asubmit(self, func, *args, **kwargs) -> asyncio.Future:
        ...

    @property
    def running_task_count(self) -> int:
        ...

    @property
    def is_full(self) -> bool:
        ...

    @property
    def is_closed(self) -> bool:
        ...

    @property
    def max_workers(self) -> int:
        ...

    @property
    def is_cpubound(self) -> bool:
        ...


class PExecutable(Protocol):
    def __executor__(self) -> PExecutor:
        ...
