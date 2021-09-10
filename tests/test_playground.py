class SyncAsync:
    def __init__(self, sync_func, async_func, *args, **kwargs) -> None:
        self.sync_func = sync_func
        self.async_func = async_func
        self.args = args
        self.kwargs = kwargs

    def __iter__(self):
        return self.sync_func(*self.args, **self.kwargs)

    def __aiter__(self):
        return self.async_func(*self.args, **self.kwargs)


from asyncio import sleep as asleep
from time import sleep


def sleep_sync(self, seconds):
    for elm in self:
        yield elm
        sleep(seconds)


async def sleep_async(self, seconds):
    for elm in self:
        yield elm
        await asleep(seconds)


def test_sleep():
    pass

    # iterable = SyncAsync(sleep_sync, sleep_async, [1, 2, 3], 1)
    # for i in iterable:
    #     print(i)

    # async def func():
    #     async for i in iterable:
    #         print(i)

    # import asyncio

    # asyncio.run(func())
