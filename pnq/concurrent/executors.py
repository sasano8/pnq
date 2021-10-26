import asyncio
import logging
from collections import deque
from concurrent.futures import Future
from concurrent.futures import ProcessPoolExecutor as _ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor as _ThreadPoolExecutor
from contextlib import AsyncExitStack, ExitStack
from functools import lru_cache, partial
from typing import TYPE_CHECKING, Union

from . import tools
from .protocols import Executor, PExecutable, PExecutor


@lru_cache(32)
def is_coroutine_function(func):
    # TODO: python3.8からはpartialが自動でasync functionを認識するので削除する
    if isinstance(func, partial):
        target = func.func
    else:
        target = func

    return asyncio.iscoroutinefunction(target)


class OverrideExecutor:
    def __executor__(self):
        return self

    @property
    def _loop(self):
        return self._get_loop()

    _get_loop = asyncio.get_running_loop

    async def __aenter__(self):
        loop = asyncio.get_running_loop()
        self._get_loop = lambda: loop
        result = super().__enter__()
        return result

    async def __aexit__(self, *args, **kwargs):
        return super().__exit__(*args, **kwargs)

    if TYPE_CHECKING:

        def submit(self, *args, **kwargs):
            ...

    def submit_async(self, func, *args, **kwargs):
        return self.submit(self._submit_async, func, *args, **kwargs)

    @staticmethod
    def _submit_async(func, *args, **kwargs):
        coro = func(*args, **kwargs)
        return asyncio.run(coro)

    def asubmit(self, func, *args, **kwargs):
        future = self.submit(func, *args, **kwargs)
        return asyncio.wrap_future(future, loop=self._loop)

    def asubmit_async(self, func, *args, **kwargs):
        future = self.submit_async(func, *args, **kwargs)
        return asyncio.wrap_future(future, loop=self._loop)

    def request(self, func, *args, **kwargs):
        if asyncio.iscoroutinefunction(func):
            future = self.submit_async(func, *args, **kwargs)
        else:
            future = self.submit(func, *args, **kwargs)
        return future

    def arequest(self, func, *args, **kwargs):
        if asyncio.iscoroutinefunction(func):
            future = self.asubmit_async(func, *args, **kwargs)
        else:
            future = self.asubmit(func, *args, **kwargs)
        return future


def get_executor(executable: Union[Executor, PExecutable]) -> PExecutor:
    if hasattr(executable, "__executor__"):
        return executable.__executor__()  # type: ignore

    return ExecutorWrapper(executable)


def asyncio_run(coroutine_function, *args, **kwargs):
    return asyncio.run(coroutine_function(*args, **kwargs))


class ExecutorWrapper(PExecutor, PExecutable):
    ASYNC_RUNNER = asyncio_run

    def __init__(self, executor: Executor):
        self.__post_init__(executor)

    def __post_init__(self, executor: Executor):
        self.executor = executor

        if isinstance(executor, _ProcessPoolExecutor):
            self._is_closed = lambda: executor._shutdown_thread  # type: ignore
            self._is_cpubound = True
            self._is_async_only = False
        elif isinstance(executor, _ThreadPoolExecutor):
            self._is_closed = lambda: executor._shutdown
            self._is_cpubound = False
            self._is_async_only = False
        else:
            raise TypeError()

        try:
            self._loop = asyncio.get_running_loop()
        except Exception:
            ...

    def __executor__(self) -> PExecutor:
        return self

    @property
    def running_task_count(self) -> int:
        raise NotImplementedError()

    @property
    def is_full(self) -> bool:
        raise NotImplementedError()

    @property
    def is_closed(self) -> bool:
        return self._is_closed()

    @property
    def max_workers(self) -> int:
        return self.executor._max_workers

    @property
    def is_cpubound(self) -> bool:
        return self._is_cpubound

    @property
    def is_async_only(self):
        return self._is_async_only

    def submit(self, func, *args, **kwargs):
        if is_coroutine_function(func):
            return self.executor.submit(
                self.__class__.ASYNC_RUNNER, func, *args, **kwargs
            )
        else:
            return self.executor.submit(func, *args, **kwargs)

    def asubmit(self, func, *args, **kwargs):
        loop = self._loop
        future = self.submit(func, *args, **kwargs)
        return asyncio.wrap_future(future, loop=loop)

    def map(self, iterable, func, unpack="", timeout=None):
        return tools.map(self, iterable, func, unpack, timeout=timeout)


class PoolContext:
    POOL_FACTORY = None
    POOL_WRAPEER = lambda x: x

    def __init__(self, max_workers=None, *args, **kwargs):
        self.args = (max_workers, *args)
        self.kwargs = kwargs
        self.executor = None

    def __enter__(self) -> PExecutor:
        if self.executor:
            raise RuntimeError()

        with ExitStack() as stack:
            executor = self.POOL_FACTORY(*self.args, **self.kwargs)
            stack.enter_context(executor)
            executor_wrapper = self.POOL_WRAPEER(executor)
            self.executor = executor
            stack.pop_all()
        return executor_wrapper

    async def __aenter__(self) -> PExecutor:
        if self.executor:
            raise RuntimeError()

        async with AsyncExitStack() as stack:
            executor = self.POOL_FACTORY(*self.args, **self.kwargs)
            stack.enter_async_context(executor)
            executor_wrapper = self.POOL_WRAPEER(executor)
            self.executor = executor
            stack.pop_all()
        return executor_wrapper

    def __exit__(self, *args, **kwargs):
        return self.executor.__exit__(*args, **kwargs)

    async def __aexit__(self, *args, **kwargs):
        return await self.executor.__aexit__(*args, **kwargs)

    @property
    def is_async_only(self):
        raise NotImplementedError()


class ProcessPoolExecutor(PoolContext):
    POOL_FACTORY = _ProcessPoolExecutor  # type: ignore
    POOL_WRAPEER = ExecutorWrapper

    async def __aenter__(self):
        return self.__enter__()

    async def __aexit__(self, *args, **kwargs):
        return self.__exit__(*args, **kwargs)


class ThreadPoolExecutor(PoolContext):
    POOL_FACTORY = _ThreadPoolExecutor  # type: ignore
    POOL_WRAPEER = ExecutorWrapper

    async def __aenter__(self):
        return self.__enter__()

    async def __aexit__(self, *args, **kwargs):
        return self.__exit__(*args, **kwargs)


class NestedFuture(asyncio.Future):
    def __init__(self, is_async, func, args, kwargs, loop, executor=None):
        super().__init__(loop=loop)
        self._task = None
        self._start_cancelled = False

        if is_async:
            self._cf = partial(func, *args, **kwargs)
        else:
            f = partial(func, *args, **kwargs)
            self._cf = partial(self._run_in_executor, loop, executor, f)

        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.executor = executor

    @staticmethod
    async def _run_in_executor(loop, executor, func):
        future = loop.run_in_executor(executor, func)
        return await future

    def start(self):
        if self.cancelled():
            return

        if self._task:
            return

        task = self._loop.create_task(self._cf())
        self._task = task
        task.add_done_callback(self._complete)

    def _complete(self, task):
        try:
            result = task.result()
            self.set_result(result)
        except BaseException as e:
            self.set_exception(e)

    @property
    def is_running(self):
        if self._task is None:
            return False

        return not self._task.done()

    def cancel(self):
        self._start_cancelled = True

        if self._task:
            self._task.cancel()
        else:
            self._super_cancel()

    def _super_cancel(self):
        super().cancel()

    def __str__(self):
        info = {
            "func": self.func.__name__,
            "args": self.args,
            "kwargs": self.kwargs,
            "state": self._state,
        }
        return f"{info}"


class AsyncPoolExecutor(PExecutor, PExecutable):
    def __init__(self, max_workers: int):
        self._max_workers = max_workers
        self._tasks = set()
        self._pre_queue = deque()
        self.is_running = False
        self._is_closed = False

        self._loop: asyncio.BaseEventLoop = None  # type: ignore
        self._threadpool: _ThreadPoolExecutor = None  # type: ignore
        self._consumer: asyncio.Task = None  # type: ignore

    def __executor__(self) -> PExecutor:
        return self

    @property
    def is_closed(self):
        return self._is_closed

    def __enter__(self):
        raise NotImplementedError()

    def __exit__(self, *args, **kwargs):
        raise NotImplementedError()

    async def __aenter__(self) -> PExecutor:
        if self._loop:
            raise RuntimeError("already initialized.")

        with ExitStack() as stack:
            self._loop = asyncio.get_event_loop()  # type: ignore
            self._threadpool = stack.enter_context(
                _ThreadPoolExecutor(self.max_workers)
            )
            self._consumer = self._loop.create_task(self.consume())
            stack.pop_all()

        self.is_running = True
        return self

    async def __aexit__(self, *args, **kwargs):
        await self.aclose()

    async def aclose(self):
        self._is_closed = True
        tasks = self._tasks

        # キューイングされた全てのタスクがスケジュールされるまで待機する
        while len(self._pre_queue):
            await asyncio.sleep(0.05)

        self._consumer.cancel()

        # 実行中のタスクを完了まで待機する
        while len(tasks):
            await asyncio.sleep(0.05)

        self._threadpool.shutdown()
        self.is_running = False

    @property
    def is_full(self):
        exists = self.max_workers <= len(self._tasks)
        return exists

    def _add_future(self, future):
        self._tasks.add(future)

    def _remove_future(self, future):
        try:
            self._tasks.remove(future)
        except Exception:
            pass

    def append_to_queue(self, item):
        self._pre_queue.append(item)

    async def consume(self):
        try:
            queue = self._pre_queue

            while True:
                while self.is_full:
                    if not self.is_running:
                        break
                    await asyncio.sleep(0.05)

                if not self.is_running:
                    break

                while True:
                    try:
                        future: NestedFuture = queue.popleft()
                        break
                    except IndexError:
                        await asyncio.sleep(0.05)

                future.start()
                self._add_future(future)
                future.add_done_callback(self._remove_future)
        except asyncio.CancelledError:
            ...
        except BaseException as e:
            logging.exception(str(e))
            raise

    def _asubmit_inner(self, is_async, func, args=None, kwargs=None):
        args = args or tuple()
        kwargs = kwargs or {}
        if self.is_closed:
            raise RuntimeError("cannot schedule new futures after shutdown")

        future = NestedFuture(
            is_async, func, args, kwargs, self._loop, self._threadpool
        )
        self.append_to_queue(future)
        return future

    def submit(self, func, *args, **kwargs):
        raise NotImplementedError()

    def asubmit(self, func, *args, **kwargs):
        is_async = is_coroutine_function(func)
        return self._asubmit_inner(is_async, func, args, kwargs)

    def amap(self, func, iterable):
        return self._amap(func, iterable)

    def _amap(self, func, iterable):
        if isinstance(func, partial):
            target = func.args[0]
        else:
            target = func

        if asyncio.iscoroutinefunction(target):

            async def main():
                return await asyncio.gather(*(func(x) for x in iterable))

            return asyncio.create_task(main())
        else:
            return self._asubmit_inner(False, map, (func, iterable))

    @property
    def running_task_count(self):
        return len(self._tasks)

    def map(self, iterable, func, unpack="", timeout=None):
        return tools.map(self, iterable, func, unpack, timeout=timeout)

    @property
    def is_async_only(self):
        return True

    @property
    def max_workers(self):
        return self._max_workers

    @property
    def is_cpubound(self):
        return False


async def exec_func(loop, func):
    return await loop.run_in_executor(None, func)


async def exec_func_async(sem, func):
    while sem._value == 0:
        await asyncio.sleep(0)

    with sem:
        return await func()


class DummyPoolExecutor(OverrideExecutor):
    def __init__(self, limit):
        import threading

        self.max_workers = limit
        self._shutdown = False
        self._sem = threading.Semaphore(limit)

    def __executor__(self) -> PExecutor:
        return self  # type: ignore

    @property
    def is_async_only(self):
        return False

    def submit(self, func, *args, **kwargs):
        if self._shutdown:
            raise RuntimeError("cannot schedule new futures after shutdown")

        f = Future()

        try:
            if is_coroutine_function(func):
                result = asyncio.run(func(*args, **kwargs))
            else:
                result = func(*args, **kwargs)

        except BaseException as e:
            f.set_exception(e)
        else:
            f.set_result(result)

        return f

    def asubmit(self, func, *args, **kwargs):
        if self._shutdown:
            raise RuntimeError("cannot schedule new futures after shutdown")

        if is_coroutine_function(func):
            async_func = partial(func, *args, **kwargs)
        else:
            async_func = partial(
                exec_func, asyncio.get_running_loop(), partial(func, *args, **kwargs)
            )
        return asyncio.create_task(exec_func_async(self._sem, async_func))

    def shutdown(self, wait=True):
        self._shutdown = True

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        ...

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args, **kwargs):
        ...

    @property
    def is_cpubound(self):
        return False


async def main():
    loop = asyncio.get_event_loop()
    loop.run_in_executor
