import asyncio
from decimal import Decimal, InvalidOperation
from typing import Any, AsyncIterable, Callable, NoReturn, Sequence, TypeVar, Union

from ..common import Listable, name_as

T = TypeVar("T")


def diter():
    ...


# async def to(source, finalizer):
#     return finalizer(await Listable(source, None))


@name_as("len")
async def _len(source: AsyncIterable[T]) -> int:
    count = 0
    async for x in Listable(source, None):
        count += 1

    return count


async def exists(source: AsyncIterable[T]) -> bool:
    async for x in source:
        return True

    return False


@name_as("all")
async def _all(source: AsyncIterable[T], selector=None) -> bool:
    async for x in Listable(source, selector):
        if not x:
            return False

    return True


@name_as("any")
async def _any(source: AsyncIterable[T], selector=None) -> bool:
    async for x in Listable(source, selector):
        if x:
            return True

    return False


async def contains(source: AsyncIterable[T], value, selector=None) -> bool:
    async for val in Listable(source, selector):
        if val == value:
            return True

    return False


@name_as("sum")
async def _sum(source: AsyncIterable[T], selector=None):
    current = 0
    async for val in Listable(source, selector):
        current += val

    return current


@name_as("min")
async def _min(source: AsyncIterable[T], selector=None, default=NoReturn):
    if default is NoReturn:
        return min(await Listable(source, selector))
    else:
        return min(await Listable(source, selector), default=default)


@name_as("max")
async def _max(source: AsyncIterable[T], selector=None, default=NoReturn):
    if default is NoReturn:
        return max(await Listable(source, selector))
    else:
        return max(await Listable(source, selector), default=default)


TH_ROUND = None


async def average(
    self: AsyncIterable[T],
    selector=lambda x: x,
    exp: float = 0.00001,
    round: TH_ROUND = "ROUND_HALF_UP",
) -> Union[float, Decimal]:
    # import statistics
    # return statistics.mean(pmap(self, selector))  # type: ignore

    seed = Decimal("0")
    i = 0
    val = 0

    async for val in Listable(self, selector):
        i += 1
        # val = selector(val)
        try:
            val = Decimal(str(val))  # type: ignore
        except InvalidOperation:
            raise TypeError(f"{val!r} is not a number")
        seed += val

    if i:
        result = seed / i
    else:
        result = seed

    result = result.quantize(Decimal(str(exp)), rounding=round)

    if isinstance(val, (int, float)):
        return float(result)
    else:
        return result


TH_ASSIGN_OP = None


async def reduce(
    self: AsyncIterable[T],
    seed: T,
    op: Union[TH_ASSIGN_OP, Callable[[Any, Any], Any]] = "+=",
    selector=lambda x: x,
) -> T:
    if callable(op):
        binary_op = op
    else:
        binary_op = MAP_ASSIGN_OP[op]

    async for val in Listable(self, selector):
        seed = binary_op(seed, val)

    return seed


async def concat(self, selector=None, delimiter: str = ""):
    to_str = lambda x: "" if x is None else str(x)  # noqa
    return delimiter.join(await Listable(Listable(self, selector), to_str))


async def each(source: AsyncIterable[T], func=lambda x: x, unpack=""):
    typ, it = diter(source)

    if typ == 0:
        if asyncio.iscoroutinefunction(func):
            for elm in source:
                await func(elm)
        else:
            for elm in source:
                func(elm)
    elif typ == 1:
        if asyncio.iscoroutinefunction(func):
            async for elm in source:
                await func(elm)
        else:
            async for elm in source:
                func(elm)


async def one(self: AsyncIterable[T]):
    it = self.__aiter__()
    try:
        result = await it.__anext__()
    except StopAsyncIteration:
        raise NoElementError()

    try:
        await it.__anext__()
        raise NotOneElementError()
    except StopAsyncIteration:
        pass

    return result


async def first(self: AsyncIterable[T]):
    it = self.__aiter__()
    try:
        return await it.__anext__()
    except StopAsyncIteration:
        raise NoElementError()


async def last(self: AsyncIterable[T]):
    if isinstance(self, Sequence):
        try:
            return self[-1]
        except IndexError:
            raise NoElementError()

    undefined = object()
    last = undefined
    async for elm in self:
        last = elm

    if last is undefined:
        raise NoElementError()
    else:
        return last


async def one_or(self: AsyncIterable[T], default):
    try:
        return await one(self)
    except NoElementError:
        return default


async def first_or(self: AsyncIterable[T], default):
    try:
        return await first(self)
    except NoElementError:
        return default


async def last_or(self: AsyncIterable[T], default):
    try:
        return await last(self)
    except NoElementError:
        return default


async def one_or_raise(self: AsyncIterable[T], exc: Union[str, Exception]):
    undefined = object()
    result = await one_or(self, undefined)
    if result is undefined:
        if isinstance(exc, str):
            raise Exception(exc)
        else:
            raise exc
    else:
        return result


async def first_or_raise(self: AsyncIterable[T], exc: Union[str, Exception]):
    undefined = object()
    result = await first_or(self, undefined)
    if result is undefined:
        if isinstance(exc, str):
            raise Exception(exc)
        else:
            raise exc
    else:
        return result


async def last_or_raise(self: AsyncIterable[T], exc: Union[str, Exception]):
    undefined = object()
    result = await last_or(self, undefined)
    if result is undefined:
        if isinstance(exc, str):
            raise Exception(exc)
        else:
            raise exc
    else:
        return result
