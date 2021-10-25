import asyncio
from typing import Any, Callable, Coroutine, TypeVar

__all__ = ["CancelToken", "run", "get_cancel_token"]


R = TypeVar("R")


class CancelToken:
    def __init__(self):
        self.is_cancelled = False

    def is_running(self):
        return not self.is_cancelled

    def cancel(self):
        self.is_cancelled = True


def run(
    func: Callable[..., Coroutine[Any, Any, R]],
    handle_signals={"SIGINT", "SIGTERM"},
) -> R:
    import asyncio
    import inspect
    import signal
    from functools import partial

    cancel_count = 0

    def handle_cancel(signame, task, token=None):
        if token is None:
            print(
                f"Cancel requested by {signame}. The task will be forcibly canceled."  # noqa
            )
            task.cancel()
        else:
            print(
                f"Cancel requested by {signame}. The task will be safely shut down. If you want to force cancel, please cancel 5 times."
            )
            token.cancel()

            nonlocal cancel_count
            cancel_count += 1

            if cancel_count >= 5:
                task.cancel()
            if cancel_count >= 10:
                exit(1)

    signature = inspect.signature(func)

    token = CancelToken()
    if len(signature.parameters) == 1:
        args = [token]
    else:
        args = []

    loop = asyncio.new_event_loop()
    set_cancel_token(loop, token)
    future = asyncio.shield(func(*args), loop=loop)

    for sig_name in handle_signals:
        sig = getattr(signal, sig_name)
        loop.add_signal_handler(sig, partial(handle_cancel, sig_name, future, *args))

    try:
        loop.run_until_complete(future)
    finally:
        loop.close()

    return future.result()


def set_cancel_token(loop, token):
    loop._cancel_token = token


def get_cancel_token():
    loop = asyncio.get_running_loop()

    token = getattr(loop, "_cancel_token", None)
    if token is None:
        raise RuntimeError("No cancel token found. please start by 'pnq.aio.run(func)'")

    return token


def cancel():
    token = get_cancel_token()
    token._is_cancelled = True
