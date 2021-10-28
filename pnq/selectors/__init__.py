import asyncio
from concurrent.futures import Future as ConcurrentFuture
from functools import cmp_to_key
from functools import partial
from functools import partial as _partial
from operator import attrgetter, itemgetter
from typing import Any, Awaitable, NoReturn

from typing_extensions import Literal

from pnq.inspect import is_coroutine_function

__all__ = [
    "starmap",
    "select",
    "select_from_attr",
    "select_from_item",
    "select_as_tuple",
    "select_as_dict",
    "select_as_awaitable",
    "select_as_future",
    "reflect",
    "flat_recursive",
    "cmp_to_key",
]


def to_str(x):
    return "" if x is None else str(x)


def _star1map(func, val):
    return func(*val)


def _star2map(func, val):
    return func(**val)


def _star3map(func, val):
    return func(*val.args, **val.kwargs)


async def _star1map_async(func, val):
    return await func(*val)


async def _star2map_async(func, val):
    return await func(**val)


async def _star3map_async(func, val):
    return await func(*val.args, **val.kwargs)


def star1map(func):
    if is_coroutine_function(func):
        return _partial(_star1map_async, func)
    else:
        return _partial(_star1map, func)


def star2map(func):
    if is_coroutine_function(func):
        return _partial(_star2map_async, func)
    else:
        return _partial(_star2map, func)


def star3map(func):
    if is_coroutine_function(func):
        return _partial(_star3map_async, func)
    else:
        return _partial(_star3map, func)


def starmap(func, unpack: Literal["", "*", "**", "***"] = "*"):
    if func is None:
        raise TypeError("func is None")

    func = to_str if func is str else func

    if unpack == "":
        return func
    elif unpack == "*":
        return star1map(func)
    elif unpack == "**":
        return star2map(func)
    elif unpack == "***":
        return star3map(func)
    else:
        raise ValueError(f"Unsupported unpack mode: {unpack}")


def select_from_attr(*args):
    return attrgetter(*args)


def select_from_item(*args):
    return itemgetter(*args)


def select(*args, attr=False):
    if attr:
        return select_from_attr(*args)
    else:
        return select_from_item(*args)


def select_as_tuple(*args, attr=False):
    if len(args) == 0:
        return lambda x: ()

    if attr:
        getter = select_from_attr(*args)
    else:
        getter = select_from_item(*args)

    if len(args) == 1:
        return lambda x: (getter(x),)
    else:
        return getter


def select_as_dict(*fields, attr=False, default=NoReturn):
    if attr:
        if default is NoReturn:
            selector = lambda x: {k: getattr(x, k) for k in fields}  # noqa
        else:
            selector = lambda x: {k: getattr(x, k, default) for k in fields}  # noqa
    else:
        if default is NoReturn:
            selector = lambda x: {k: x[k] for k in fields}  # noqa
        else:

            def getitem(obj, k):
                try:
                    return obj[k]
                except Exception:
                    return default

            selector = lambda x: {k: getitem(x, k) for k in fields}  # noqa

    return selector


def reflect(mapping, attr: bool = False):
    transposed = _transpose(mapping)
    single, multi = _split_single_multi(transposed)
    return _build_selector(single, multi, attr)


def _transpose(mapping):
    from collections import defaultdict

    tmp = defaultdict(list)

    for left, right in mapping.items():
        if isinstance(right, str):
            tmp[left].append(right)
        elif isinstance(right, list):
            tmp[left] = right
        elif isinstance(right, tuple):
            tmp[left] = right
        elif isinstance(right, set):
            tmp[left] = right
        else:
            raise TypeError(f"{v} is not a valid mapping")

    # output属性 - 元の属性（複数の場合あり）
    target = defaultdict(list)

    for k, outputs in tmp.items():
        for out in outputs:
            target[out].append(k)

    return target


def _split_single_multi(dic):
    single = {}
    multi = {}
    for k, v in dic.items():
        if len(v) > 1:
            multi[k] = v
        else:
            single[k] = v[0]

    return single, multi


def _build_selector(single, multi, attr: bool = False):
    if attr:
        getter = getattr

    else:
        getter = dict.__getitem__

    def reflector(x):
        result = {}
        for k, v in single.items():
            result[k] = getter(x, v)

        for k, fields in multi.items():
            result[k] = []
            for f in fields:
                result[k].append(getter(x, f))

        return result

    return reflector


def flat_recursive(func):
    def wrapper(x):
        yield x
        try:
            nodes = func(x)
            for node in nodes:
                yield from wrapper(node)

        except Exception:
            ...

    return wrapper


async def _awaitable_wrapper(awaitable):
    return await awaitable


def _select_future_with_schedule(target) -> asyncio.Future:
    if hasattr(target, "__await__"):
        if asyncio.isfuture(target):
            return target
        elif asyncio.iscoroutine(target):
            return asyncio.create_task(target)
        else:
            aw = _awaitable_wrapper(target)
            return asyncio.create_task(aw)

    if isinstance(target, ConcurrentFuture):
        return asyncio.wrap_future(target)

    raise TypeError(f"{target} is not awaitable")
