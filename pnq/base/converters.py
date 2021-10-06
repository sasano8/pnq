import asyncio
from typing import Iterator, TypeVar

T = TypeVar("T")


class QuerySyncToAsync(Query[T]):
    """同期イテレータを非同期イテレータに変換します。
    もしくは、同期イテレータを取得できない場合、非同期イテレータの取得を試みます。"""

    iter_type = IterType.ASYNC

    def __init__(self, source: Iterable[T]):
        # super().__init__(self)
        self.source = source
        self.run_iter_type = self.iter_type

    def _impl_iter(self):
        raise NotImplementedError()

    def _impl_aiter(self):
        it = None
        try:
            it = iter(self.source)
        except Exception:
            pass

        if it:
            return sync_to_async_iterator(it)
        else:
            return self.source.__aiter__()


class ToSync:
    """非同期イテレータを同期イテレータに変換する"""

    iter_type = IterType.SYNC

    def __iter__(self) -> Iterator[T]:
        if self.loop:
            loop = self.loop
        else:
            loop = asyncio.new_event_loop()  # +python3.8

        try:
            aiter = self.source.__aiter__()
        except Exception:
            loop.close()

        async def get_next(aiter):
            try:
                obj = await aiter.__anext__()
                return False, obj
            except StopAsyncIteration:
                return True, None

        try:
            while True:
                done, obj = loop.run_until_complete(get_next(aiter))
                if done:
                    break
                yield obj
        finally:
            loop.close()
