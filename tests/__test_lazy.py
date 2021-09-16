from asyncio import coroutine
from typing import Any, Awaitable, Generator


class LazyAction(Awaitable[int]):
    def __init__(self, src, action, *args, **kwargs):
        self.src = src
        self.action = action
        self.args = args
        self.kwargs = kwargs

    def __call__(self):
        return self.action(self.src, *self.args, **self.kwargs)

    async def __acall__(self):
        it = self.__aiter_action__()
        return await self.action(it, *self.args, **self.kwargs)

    def __await__(self) -> Generator[Any, Any, int]:
        return self.__acall__().__await__()

    def __iter_action__(self):
        yield from self.src

    async def __aiter_action__(self):
        for v in self.src:
            yield v

    async def to_coro(self):
        return await self.__acall__()


def test_lazy():
    import asyncio

    async def main():
        obj = LazyAction([1], list)
        assert obj() == [1]
        assert await obj == 5
        assert list(obj) == [1, 2, 3]
        assert [x async for x in obj] == [4, 5, 6]

    # assert asyncio.run(Lazy().to_coro()) == 5
    asyncio.run(main())


"""
ケース

イテレータからクラスを生成するタイプ
イテレータを受け取ってループ処理する関数

list(aiter) # NG
[x async for x in aiter]
afunc(aiter)
await pnq.list(aiter)

await pnq.query([]).lazy(pnq.list)

"""
