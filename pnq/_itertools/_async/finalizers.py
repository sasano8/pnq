import asyncio
from decimal import Decimal, InvalidOperation
from typing import Any, AsyncIterable, Callable, NoReturn, Sequence, TypeVar, Union

from pnq.exceptions import NoElementError, NotOneElementError

from ..common import Listable, name_as
from ..op import MAP_ASSIGN_OP, TH_ASSIGN_OP, TH_ROUND

T = TypeVar("T")


async def to(source, finalizer):
    return await finalizer(source)


@name_as("len")
async def _len(source: AsyncIterable[T]) -> int:
    count = 0
    async for x in Listable(source, None):
        count += 1

    return count


async def empty(source: AsyncIterable[T]) -> bool:
    async for x in source:
        return False

    return True


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


# containsより意思的
async def find(source: AsyncIterable[T], value, selector=None) -> bool:
    async for val in Listable(source, selector):
        if val == value:
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
async def _min(source: AsyncIterable[T], key_selector=None, default=NoReturn):
    if default is NoReturn:
        return min(await Listable(source), key=key_selector)
    else:
        return min(await Listable(source), key=key_selector, default=default)


@name_as("max")
async def _max(source: AsyncIterable[T], key_selector=None, default=NoReturn):
    if default is NoReturn:
        return max(await Listable(source), key=key_selector)
    else:
        return max(await Listable(source), key=key_selector, default=default)


async def average(
    source: AsyncIterable[T],
    selector=lambda x: x,
    exp: float = 0.00001,
    round: TH_ROUND = "ROUND_HALF_UP",
) -> Union[float, Decimal]:
    # import statistics
    # return statistics.mean(pmap(self, selector))  # type: ignore

    seed = Decimal("0")
    i = 0
    val = 0

    async for val in Listable(source, selector):
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


async def reduce(
    source: AsyncIterable[T],
    seed: Union[T, None],
    op: Union[TH_ASSIGN_OP, Callable[[Any, Any], Any]] = "+=",
    selector=lambda x: x,
) -> T:
    if isinstance(op, str):
        binary_op = MAP_ASSIGN_OP[op]
    else:
        binary_op = op

    it = Listable(source, selector).__aiter__()

    if seed is None:
        async for val in it:
            seed = val
            break
        else:
            raise TypeError("empty sequence with no seed.")

    async for val in Listable(source, selector):
        seed = binary_op(seed, val)

    return seed  # type: ignore


async def accumulate(
    source: AsyncIterable[T],
    seed: Union[T, None],
    op: Union[TH_ASSIGN_OP, Callable[[Any, Any], Any]] = "+=",
    selector=lambda x: x,
):
    if isinstance(op, str):
        binary_op = MAP_ASSIGN_OP[op]
    else:
        binary_op = op

    it = Listable(source, selector).__aiter__()

    results = []

    if seed is None:
        async for val in it:
            seed = val
            results.append(seed)
            break
        else:
            return results
    else:
        results.append(seed)

    async for val in it:
        seed = binary_op(seed, val)
        results.append(seed)

    return results


async def concat(source: AsyncIterable[T], selector=None, delimiter: str = ""):
    to_str = lambda x: "" if x is None else str(x)  # noqa
    return delimiter.join(await Listable(Listable(source, selector), to_str))


async def each(source: AsyncIterable[T], func=lambda x: x, unpack=""):
    if asyncio.iscoroutinefunction(func):
        async for elm in source:
            await func(elm)
    else:
        async for elm in source:
            func(elm)


def each_unpack(source: AsyncIterable[T], func):
    return each(source, func, unpack="*")


async def each_async(*args, **kwargs):
    ...


async def each_async_unpack(*args, **kwargs):
    ...


async def one(source: AsyncIterable[T]):
    it = source.__aiter__()
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


async def first(source: AsyncIterable[T]):
    it = source.__aiter__()
    try:
        return await it.__anext__()
    except StopAsyncIteration:
        raise NoElementError()


async def last(source: AsyncIterable[T]):
    if isinstance(source, Sequence):
        try:
            return source[-1]
        except IndexError:
            raise NoElementError()

    undefined = object()
    last = undefined
    async for elm in source:
        last = elm

    if last is undefined:
        raise NoElementError()
    else:
        return last


async def one_or(source: AsyncIterable[T], default):
    try:
        return await one(source)
    except NoElementError:
        return default


async def first_or(source: AsyncIterable[T], default):
    try:
        return await first(source)
    except NoElementError:
        return default


async def last_or(source: AsyncIterable[T], default):
    try:
        return await last(source)
    except NoElementError:
        return default


async def one_or_raise(source: AsyncIterable[T], exc: Union[str, Exception]):
    undefined = object()
    result = await one_or(source, undefined)
    if result is undefined:
        if isinstance(exc, str):
            raise Exception(exc)
        else:
            raise exc
    else:
        return result


async def first_or_raise(source: AsyncIterable[T], exc: Union[str, Exception]):
    undefined = object()
    result = await first_or(source, undefined)
    if result is undefined:
        if isinstance(exc, str):
            raise Exception(exc)
        else:
            raise exc
    else:
        return result


async def last_or_raise(source: AsyncIterable[T], exc: Union[str, Exception]):
    undefined = object()
    result = await last_or(source, undefined)
    if result is undefined:
        if isinstance(exc, str):
            raise Exception(exc)
        else:
            raise exc
    else:
        return result
