from typing import Any, Callable, Coroutine, TypeVar

from pnq.aio import run

from .core import PnqInternalDict, PnqInternalSeq, PnqInternalSet, QueryNode

R = TypeVar("R")


class Builder:
    QUERY = QueryNode
    QUERY_SEQ = PnqInternalSeq
    QUERY_DICT = PnqInternalDict
    QUERY_SET = PnqInternalSet

    @classmethod
    def query(cls, source):
        if isinstance(source, list):
            return cls.QUERY_SEQ(source)
        elif isinstance(source, dict):
            return cls.QUERY_DICT(source)
        elif isinstance(source, tuple):
            return cls.QUERY_SEQ(source)
        elif isinstance(source, set):
            return cls.QUERY_SET(source)
        elif isinstance(source, frozenset):
            return cls.QUERY_SET(source)
        elif isinstance(source, QueryNode):
            return source
        else:
            return cls.QUERY(source)

    def infinite(func, *args, **kwargs):
        def infinite(*args, **kwargs):
            while True:
                yield func(*args, **kwargs)

        return query(LazyGenerator(infinite, *args, **kwargs))

    def count(start=0, step=1):
        from itertools import count

        return query(LazyGenerator(count, start, step))

    def cycle(iterable):
        from itertools import cycle

        return query(LazyGenerator(cycle, iterable))

    @classmethod
    def run(
        cls,
        func: Callable[..., Coroutine[Any, Any, R]],
        handle_signals={"SIGINT", "SIGTERM"},
    ) -> R:
        return run(func, handle_signals)
