import asyncio
from contextlib import AsyncExitStack, ExitStack
from typing import TYPE_CHECKING, Iterable


class Arguments:
    """引数を保持するためのクラスです。

    Usage:
    ```
    >>> def show(a1, a2, kw1, kw2):
    >>>   print(a1, a2, kw1, kw2)
    >>> a = pnq.args(1, 2, kw1=3, kw2=4)
    >>> pnq.query([a]).each(show, unpack="***")
    1, 2, 3, 4
    ```
    """

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __iter__(self):
        return iter(self.args)

    def __getitem__(self, key):
        return self.kwargs[key]

    def keys(self):
        return self.kwargs.keys()

    def values(self):
        return self.kwargs.values()

    def items(self):
        return self.kwargs.items()

    @classmethod
    def from_obj(cls, args, kwargs):
        obj = cls()
        obj.args = args
        obj.kwargs = kwargs
        return obj


class PnqExitStack:
    def __init__(self, contexts: Iterable):
        self.source = contexts
        self._init = False

    @classmethod
    def from_args(cls, *contexts):
        return cls(contexts)

    def __iter__(self):
        return iter(self.source)

    def start(self):
        if self._init:
            raise RuntimeError("ExitStack is already initialized")

        contexts = []
        with ExitStack() as stack:
            for item in self.source:
                stack.enter_context(item)
                contexts.append(item)

            self.close = stack.pop_all().close  # type: ignore

        self._init = True
        return tuple(contexts)

    async def astart(self):
        if self._init:
            raise RuntimeError("ExitStack is already initialized")

        contexts = []
        async with AsyncExitStack() as stack:
            for item in self.source:
                await stack.enter_async_context(item)
                contexts.append(item)

            self.aclose = stack.pop_all().aclose  # type: ignore

        self._init = True
        return tuple(contexts)

    if TYPE_CHECKING:

        def close(self):
            ...

        async def aclose(self):
            ...

    def __enter__(self):
        return self.start()

    def __exit__(self, *args, **kwargs):
        self.close()

    async def __aenter__(self):
        return await self.astart()

    async def __aexit__(self, *args, **kwargs):
        await self.aclose()


exitstack = PnqExitStack.from_args


class PnqQueue:
    def __init__(self) -> None:
        from collections import deque

        self._items = deque()
        self._running = True

    def stop(self):
        self._running = False

    def kill(self):
        self._running = False
        self._items.clear()

    def put(self, item):
        if not self._running:
            raise RuntimeError("Queue is stopped")
        self._items.append(item)

    def try_pop(self):
        if not self._running:
            raise RuntimeError("Queue is stopped")
        if not self._items:
            return False, None
        return True, self._items.popleft()

    async def __aiter__(self):
        if not self._running:
            raise RuntimeError("Queue is stopped")

        sleep = asyncio.sleep
        queue = self._items

        while self._running:
            if not queue:
                await sleep(0)
                continue
            yield queue.popleft()

        while queue:
            yield queue.popleft()
