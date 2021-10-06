from decimal import Decimal, InvalidOperation
from typing import (
    Any,
    Callable,
    Coroutine,
    Generic,
    Iterable,
    NoReturn,
    Sequence,
    TypeVar,
    Union,
)

from .exceptions import NoElementError, NotOneElementError
from .op import MAP_ASSIGN_OP, TH_ASSIGN_OP, TH_ROUND

T = TypeVar("T")


def to(self, finalizer):
    """ストリームをファイナライザによって処理します。

    Args:

    * self: 評価するシーケンス
    * finalizer: イテレータを受け取るクラス・関数

    Returns: ファイナライザが返す結果

    Usage:
    ```
    >>> pnq.query([1, 2, 3]).to(list)
    [1, 2]
    >>> pnq.query({1: "a", 2: "b"}).to(dict)
    {1: "a", 2: "b"}
    ```
    """
    return finalizer(self)


def map_if_not_none(it, selector):
    if selector is None:
        return it
    else:
        return map(selector, it)


def map_async_if_not_none(it, selector):
    if selector is None:
        return it
    else:
        return (selector(x) async for x in it)


class SyncFinalizer(Generic[T]):
    def __init__(self, source: Iterable[T]):
        self.source = source

    def __iter__(self):
        return self.source.__iter__()

    def to(self, finalizer):
        if False:  # is_async
            raise Exception()
        else:
            return finalizer(self)

    def lazy(self, finalizer):
        ...

    def len(self):
        if hasattr(self.source, "__len__"):
            return len(self.source)

        i = 0
        for i, v in enumerate(self, 1):
            ...

        return i

    def exists(self):
        for elm in self:
            return True

        return False

    def all(self, selector=lambda x: x):
        return all(map_if_not_none(self, selector))

    def any(self, selector=lambda x: x):
        return any(map_if_not_none(self, selector))

    def contains(self, value, selector=lambda x: x) -> bool:
        for val in map_if_not_none(self, selector):
            if val == value:
                return True

        return False

    def min(self, selector=lambda x: x, default=NoReturn):
        if default is NoReturn:
            return min(map_if_not_none(self, selector))
        else:
            return min(map_if_not_none(self, selector), default=default)

    def max(self, selector=lambda x: x, default=NoReturn):
        if default is NoReturn:
            return max(map_if_not_none(self, selector))
        else:
            return max(map_if_not_none(self, selector), default=default)

    def sum(self, selector=lambda x: x):
        return sum(map_if_not_none(self, selector))

    def average(
        self,
        selector=lambda x: x,
        exp: float = 0.00001,
        round: TH_ROUND = "ROUND_HALF_UP",
    ) -> Union[float, Decimal]:
        # import statistics
        # return statistics.mean(pmap(self, selector))  # type: ignore

        seed = Decimal("0")
        i = 0
        val = 0

        for i, val in enumerate(map_if_not_none(self, selector), 1):
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

    def reduce(
        self,
        seed: T,
        op: Union[TH_ASSIGN_OP, Callable[[Any, Any], Any]] = "+=",
        selector=lambda x: x,
    ) -> T:
        if callable(op):
            binary_op = op
        else:
            binary_op = MAP_ASSIGN_OP[op]

        for val in map_if_not_none(self, selector):
            seed = binary_op(seed, val)

        return seed

    def concat(self, selector=lambda x: x, delimiter: str = ""):
        to_str = lambda x: "" if x is None else str(x)  # noqa
        return delimiter.join(to_str(x) for x in map_if_not_none(self, selector))

    def each(self, func=lambda x: x):
        for elm in self:
            func(elm)

    def each_unpack(self, func):
        for elm in self:
            func(**elm)

    async def each_async(self, func):
        for elm in self:
            await func(elm)

    async def each_async_unpack(self, func):
        for elm in self:
            await func(**elm)

    def one(self):
        it = iter(self)
        try:
            result = next(it)
        except StopIteration:
            raise NoElementError()

        try:
            next(it)
            raise NotOneElementError()
        except StopIteration:
            pass

        return result

    def first(self):
        it = iter(self)
        try:
            return next(it)
        except StopIteration:
            raise NoElementError()

    def last(self):
        if isinstance(self, Sequence):
            try:
                return self[-1]
            except IndexError:
                raise NoElementError()

        undefined = object()
        last = undefined
        for elm in self:
            last = elm

        if last is undefined:
            raise NoElementError()
        else:
            return last

    def one_or(self, default):
        try:
            # return one(self)
            return self.one()
        except NoElementError:
            return default

    def first_or(self, default):
        try:
            # return first(self)
            return self.first()
        except NoElementError:
            return default

    def last_or(self, default):
        try:
            # return last(self)
            return self.last()
        except NoElementError:
            return default

    def one_or_raise(self, exc: Union[str, Exception]):
        undefined = object()
        # result = one_or(self, undefined)
        result = self.one_or(undefined)
        if result is undefined:
            if isinstance(exc, str):
                raise Exception(exc)
            else:
                raise exc
        else:
            return result

    def first_or_raise(self, exc: Union[str, Exception]):
        undefined = object()
        # result = first_or(self, undefined)
        result = self.first_or(undefined)
        if result is undefined:
            if isinstance(exc, str):
                raise Exception(exc)
            else:
                raise exc
        else:
            return result

    def last_or_raise(self, exc: Union[str, Exception]):
        undefined = object()
        # result = last_or(self, undefined)
        result = self.last_or(undefined)
        if result is undefined:
            if isinstance(exc, str):
                raise Exception(exc)
            else:
                raise exc
        else:
            return result


class AsyncFinalizer(Generic[T]):
    def __init__(self, source: Iterable[T]):
        self.source = source

    def __aiter__(self):
        return self.source.__aiter__()

    async def to(self, finalizer):
        if finalizer is list:
            return [x async for x in self]
        elif finalizer is dict:
            return {k: v async for k, v in self}
        elif finalizer is set:
            return {x async for x in self}
        else:
            return await finalizer(self)

    async def lazy(self, finalizer):
        ...

    async def len(self) -> int:
        i = 0
        async for x in self:
            i += 1

        return i

    async def exists(self) -> bool:
        async for elm in self:
            return True

        return False

    async def all(self, selector=lambda x: x) -> bool:
        async for x in map_async_if_not_none(self, selector):
            if not x:
                return False
        return True

    async def any(self, selector=lambda x: x) -> bool:
        async for x in map_async_if_not_none(self, selector):
            if x:
                return True
        return False

    async def contains(self, value, selector=lambda x: x) -> bool:
        async for val in map_async_if_not_none(self, selector):
            if val == value:
                return True

        return False

    async def min(self, selector=lambda x: x, default=NoReturn) -> T:
        it = map_async_if_not_none(self, selector).__aiter__()

        try:
            _min = await it.__anext__()
        except StopAsyncIteration:
            if default is NoReturn:
                raise
            else:
                return default

        async for x in it:
            if x < _min:
                _min = x

        return _min

    async def max(self, selector=lambda x: x, default=NoReturn) -> T:
        it = map_async_if_not_none(self, selector).__aiter__()

        try:
            _max = await it.__anext__()
        except StopAsyncIteration:
            if default is NoReturn:
                raise
            else:
                return default

        async for x in it:
            if x > _max:
                _max = x

        return _max

    async def sum(self, selector=lambda x: x):
        it = map_async_if_not_none(self, selector).__aiter__()

        try:
            _sum = await it.__anext__()
        except StopIteration:
            return 0

        async for x in it:
            _sum += x

        return _sum

    async def average(
        self,
        selector=lambda x: x,
        exp: float = 0.00001,
        round: TH_ROUND = "ROUND_HALF_UP",
    ) -> Union[float, Decimal]:
        # import statistics
        # return statistics.mean(pmap(self, selector))  # type: ignore

        seed = Decimal("0")
        i = 0
        val = 0

        async for val in map_async_if_not_none(self, selector):
            i += 1

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
        self,
        seed: T,
        op: Union[TH_ASSIGN_OP, Callable[[Any, Any], Any]] = "+=",
        selector=lambda x: x,
    ) -> T:
        if callable(op):
            binary_op = op
        else:
            binary_op = MAP_ASSIGN_OP[op]

        async for val in map_async_if_not_none(self, selector):
            seed = binary_op(seed, val)

        return seed

    async def concat(self, selector=lambda x: x, delimiter: str = "") -> str:
        to_str = lambda x: "" if x is None else str(x)  # noqa
        ait = (
            to_str(x) async for x in map_async_if_not_none(self, selector)
        ).__aiter__()

        # TODO: string.IOを使う
        try:
            current = await ait.__anext__()
        except StopAsyncIteration:
            return ""

        async for x in ait:
            current += delimiter + x

        return current

    async def each(self, func=lambda x: x) -> None:
        async for elm in self:
            func(elm)

    async def each_unpack(self, func) -> None:
        async for elm in self:
            func(**elm)

    async def each_async(self, func) -> None:
        async for elm in self:
            await func(elm)

    async def each_async_unpack(self, func) -> None:
        async for elm in self:
            await func(**elm)

    async def one(self) -> T:
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

    async def first(self) -> T:
        it = self.__aiter__()
        try:
            return await it.__anext__()
        except StopIteration:
            raise NoElementError()

    async def last(self):
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

    async def one_or(self, default):
        try:
            # return one(self)
            return await self.one()
        except NoElementError:
            return default

    async def first_or(self, default):
        try:
            # return first(self)
            return await self.first()
        except NoElementError:
            return default

    async def last_or(self, default):
        try:
            # return last(self)
            return await self.last()
        except NoElementError:
            return default

    async def one_or_raise(self, exc: Union[str, Exception]):
        undefined = object()
        # result = one_or(self, undefined)
        result = await self.one_or(undefined)
        if result is undefined:
            if isinstance(exc, str):
                raise Exception(exc)
            else:
                raise exc
        else:
            return result

    async def first_or_raise(self, exc: Union[str, Exception]):
        undefined = object()
        # result = first_or(self, undefined)
        result = await self.first_or(undefined)
        if result is undefined:
            if isinstance(exc, str):
                raise Exception(exc)
            else:
                raise exc
        else:
            return result

    async def last_or_raise(self, exc: Union[str, Exception]):
        undefined = object()
        # result = last_or(self, undefined)
        result = await self.last_or(undefined)
        if result is undefined:
            if isinstance(exc, str):
                raise Exception(exc)
            else:
                raise exc
        else:
            return result
