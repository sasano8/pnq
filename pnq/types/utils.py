import asyncio
from concurrent.futures import Future as _Future


def _wrap_future(future: _Future) -> asyncio.Future:
    _future = asyncio.wrap_future(future)
    if not (future is _future):
        _future.__future__ = lambda: future  # type: ignore
    return _future


def as_future(target) -> _Future:
    if isinstance(target, _Future):
        return target
    return target.__future__()


def as_afuture(target) -> asyncio.Future:
    if isinstance(target, asyncio.Future):
        return target
    if isinstance(target, _Future):
        return _wrap_future(target)

    if hasattr(target, "__afuture__"):
        return target.__afuture__()
    else:
        return _wrap_future(target.__future__())
