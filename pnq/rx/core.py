from contextlib import ExitStack
from functools import partial
from typing import Callable, ContextManager, Dict, Generic, List, Protocol, TypeVar

T = TypeVar("T")


class MyExitStack:
    def __init__(self, *contexts: ContextManager, no_raise: bool = False):
        self._contexts = contexts
        self._stack: ExitStack = None

    def __enter__(self):
        if self._stack:
            raise RuntimeError()

        self._stack = ExitStack().__enter__()
        try:
            for k, v in self._list_context():
                self.enter_context(v)

        except Exception as e:
            self.__exit__()

        return self

    def _list_context(self):
        yield from enumerate(self._contexts)

    def __exit__(self, *args, **kwargs):
        if not self._stack:
            return

        self._stack.__exit__(*args, **kwargs)
        self._stack = None

    def enter_context(self, cm):
        if not self._stack:
            raise RuntimeError()

        ctx = self._stack.enter_context(cm)
        return ctx


class NamedExitStack(MyExitStack):
    def __init__(self, **contexts: ContextManager):
        self._contexts = contexts
        self._stack: ExitStack = None

    def _list_context(self):
        yield from self._contexts.items()


class Subscription:
    def __init__(self, undistribuite: Callable[[], None]) -> None:
        self._undistribuite = undistribuite

    def __enter__(self) -> "Subscription":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self._undistribuite()


# Subject
class Distributer(Generic[T]):
    def __init__(self):
        self._subscribers = set()

    def distribuite(self, subscriber: "Subscriber[T]") -> Subscription:
        cancel = partial(self.undistribuite, subscriber)
        sub = Subscription(cancel)
        self._subscribers.add(subscriber)
        return sub

    def undistribuite(self, subscriber: "Subscriber[T]") -> Subscription:
        self._subscribers.discard(subscriber)

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        while len(self._subscribers):
            subscriber = self._subscribers.pop()
            self.undistribuite(subscriber)

    def on_next(self, value: T):
        for sub in self._subscribers:
            sub.on_next(value)

    def on_error(self, err: Exception):
        for sub in self._subscribers:
            sub.on_error(err)


class Subscriber(Generic[T]):
    def callback_on_next(self, func):
        self._on_next = func
        return func

    def callback_on_err(self, func):
        self._on_err = func
        return func

    def subscribe(self, distributer: "Distributer[T]") -> Subscription:
        return distributer.distribuite(self)

    def unsubscribe(self, distributer: "Distributer[T]") -> Subscription:
        return distributer.undistribuite(self)

    def on_next(self, value: T) -> None:
        self._on_next(value)

    def on_error(self, error: Exception) -> None:
        self._on_err(error)

    def on_completed(self) -> None:
        ...
