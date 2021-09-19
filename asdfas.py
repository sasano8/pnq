from typing import Any, Awaitable, Generator


class Defer(Awaitable[float]):
    def __call__(self) -> int:
        pass

    def __await__(self):
        return super().__await__()


async def main():
    r1 = Defer()()
    print(r1)

    result = await Defer()
    print(result)
