from typing import AsyncIterable, Callable, Generic, Iterable, TypeVar, Union

T = TypeVar("T")


class Listable(Iterable[T]):
    def __init__(self, iterable: Iterable[T]):
        self.source = iterable

    def __iter__(self):
        return iter(self.source)

    def save(self):
        return self.result()

    def result(self, timeout=None):
        # TODO: implement timeout
        return list(self.source)

    def __future__(self):
        raise NotImplementedError()


class AsyncListable(AsyncIterable[T]):
    def __init__(self, iterable: AsyncIterable[T]):
        self.source = iterable

    def __aiter__(self):
        return self.source.__aiter__()

    async def save_async(self):
        return [x async for x in self]

    def __await__(self):
        return self.save_async().__await__()

    def __afuture__(self):
        raise NotImplementedError()


class LazyListable(Listable[T], AsyncListable[T]):  # type: ignore
    def __init__(self, iterable: Union[Iterable[T], AsyncIterable[T]]):
        self.source = iterable  # type: ignore

    def as_iter(self) -> Listable[T]:
        return Listable(self.source)

    def as_aiter(self) -> AsyncListable[T]:
        return AsyncListable(self.source)  # type: ignore


class LazyGeneratorFunction(Generic[T]):
    def __init__(
        self, func: Union[Callable[..., Iterable[T]], Callable[..., AsyncIterable[T]]]
    ):
        self.func = func

    def __call__(self, *args, **kwargs) -> LazyListable[T]:
        return LazyListable(self.func(*args, **kwargs))
