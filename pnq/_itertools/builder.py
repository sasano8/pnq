from typing import Any, Callable, Coroutine, TypeVar

from .core import Query, QueryAsync, QueryDict, QueryNormal, QuerySeq, QuerySet

R = TypeVar("R")


class CancelToken:
    def __init__(self):
        self.is_cancelled = False

    def is_running(self, *args):
        return not self.is_cancelled

    def cancel(self):
        self.is_cancelled = True


class Builder:
    QUERY_BOTH = Query
    QUERY_ASYNC = QueryAsync
    QUERY_NORMAL = QueryNormal
    QUERY_SEQ = QuerySeq
    QUERY_DICT = QueryDict
    QUERY_SET = QuerySet

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
        elif isinstance(source, Query):
            return source
        else:
            has_iter = hasattr(source, "__iter__")
            has_aiter = hasattr(source, "__aiter__")

            if has_iter and has_aiter:
                return cls.QUERY_BOTH(source)
            elif has_iter:
                return cls.QUERY_NORMAL(source)
            elif has_aiter:
                return cls.QUERY_ASYNC(source)
            else:
                raise TypeError()

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
        import asyncio
        import inspect
        import signal
        from functools import partial

        def handle_cancel(signame, task, token=None):
            if token is None:
                print(
                    f"Cancel requested by {signame}. The task will be forcibly canceled."  # noqa
                )
                task.cancel()
            else:
                print(
                    f"Cancel requested by {signame}. The task will be safely shut down."
                )
                token.cancel()

        signature = inspect.signature(func)

        if len(signature.parameters) == 1:
            args = [CancelToken()]
        else:
            args = []

        loop = asyncio.new_event_loop()
        future = asyncio.shield(func(*args), loop=loop)

        for sig_name in handle_signals:
            sig = getattr(signal, sig_name)
            loop.add_signal_handler(
                sig, partial(handle_cancel, sig_name, future, *args)
            )

        try:
            loop.run_until_complete(future)
        finally:
            loop.close()

        return future.result()
