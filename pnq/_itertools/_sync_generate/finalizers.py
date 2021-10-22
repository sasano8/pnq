import asyncio
from decimal import Decimal, InvalidOperation
from typing import Any, Callable, Iterable, NoReturn, Sequence, TypeVar, Union

from ..common import Listable, name_as

T = TypeVar("T")


def diter():
    ...


# async def to(source, finalizer):
#     return finalizer(await Listable(source, None))


@name_as("len")
def _len(source: Iterable[T]) -> int:
    count = 0
    for x in Listable(source, None):
        count += 1

    return count


def exists(source: Iterable[T]) -> bool:
    for x in source:
        return True

    return False


@name_as("all")
def _all(source: Iterable[T], selector=None) -> bool:
    for x in Listable(source, selector):
        if not x:
            return False

    return True


@name_as("any")
def _any(source: Iterable[T], selector=None) -> bool:
    for x in Listable(source, selector):
        if x:
            return True

    return False


def contains(source: Iterable[T], value, selector=None) -> bool:
    for val in Listable(source, selector):
        if val == value:
            return True

    return False


@name_as("sum")
def _sum(source: Iterable[T], selector=None):
    current = 0
    for val in Listable(source, selector):
        current += val

    return current


@name_as("min")
def _min(source: Iterable[T], selector=None, default=NoReturn):
    if default is NoReturn:
        return min(Listable(source, selector))
    else:
        return min(Listable(source, selector), default=default)


@name_as("max")
def _max(source: Iterable[T], selector=None, default=NoReturn):
    if default is NoReturn:
        return max(Listable(source, selector))
    else:
        return max(Listable(source, selector), default=default)


TH_ROUND = None


def average(
    self: Iterable[T],
    selector=lambda x: x,
    exp: float = 0.00001,
    round: TH_ROUND = "ROUND_HALF_UP",
) -> Union[float, Decimal]:
    # import statistics
    # return statistics.mean(pmap(self, selector))  # type: ignore

    seed = Decimal("0")
    i = 0
    val = 0

    for val in Listable(self, selector):
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


def reduce(
    self: Iterable[T],
    seed: T,
    op: Union[TH_ASSIGN_OP, Callable[[Any, Any], Any]] = "+=",
    selector=lambda x: x,
) -> T:
    if callable(op):
        binary_op = op
    else:
        binary_op = MAP_ASSIGN_OP[op]

    for val in Listable(self, selector):
        seed = binary_op(seed, val)

    return seed


def concat(self, selector=None, delimiter: str = ""):
    to_str = lambda x: "" if x is None else str(x)  # noqa
    return delimiter.join(Listable(Listable(self, selector), to_str))


def each(source: Iterable[T], func=lambda x: x, unpack=""):
    typ, it = diter(source)

    if typ == 0:
        if asyncio.iscoroutinefunction(func):
            for elm in source:
                func(elm)
        else:
            for elm in source:
                func(elm)
    elif typ == 1:
        if asyncio.iscoroutinefunction(func):
            for elm in source:
                func(elm)
        else:
            for elm in source:
                func(elm)


def one(self: Iterable[T]):
    it = self.__iter__()
    try:
        result = it.__next__()
    except StopIteration:
        raise NoElementError()

    try:
        it.__next__()
        raise NotOneElementError()
    except StopIteration:
        pass

    return result


def first(self: Iterable[T]):
    it = self.__iter__()
    try:
        return it.__next__()
    except StopIteration:
        raise NoElementError()


def last(self: Iterable[T]):
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


def one_or(self: Iterable[T], default):
    try:
        return one(self)
    except NoElementError:
        return default


def first_or(self: Iterable[T], default):
    try:
        return first(self)
    except NoElementError:
        return default


def last_or(self: Iterable[T], default):
    try:
        return last(self)
    except NoElementError:
        return default


def one_or_raise(self: Iterable[T], exc: Union[str, Exception]):
    undefined = object()
    result = one_or(self, undefined)
    if result is undefined:
        if isinstance(exc, str):
            raise Exception(exc)
        else:
            raise exc
    else:
        return result


def first_or_raise(self: Iterable[T], exc: Union[str, Exception]):
    undefined = object()
    result = first_or(self, undefined)
    if result is undefined:
        if isinstance(exc, str):
            raise Exception(exc)
        else:
            raise exc
    else:
        return result


def last_or_raise(self: Iterable[T], exc: Union[str, Exception]):
    undefined = object()
    result = last_or(self, undefined)
    if result is undefined:
        if isinstance(exc, str):
            raise Exception(exc)
        else:
            raise exc
    else:
        return result
