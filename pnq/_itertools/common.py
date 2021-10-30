def name_as(name):
    def wrapper(func):
        func.__name__ = name
        return func

    return wrapper


class Listable:
    def __init__(self, source, selector=None) -> None:
        self.source = source
        self.selector = selector

    def __iter__(self):
        selector = self.selector
        it = self.source.__iter__()

        if selector is None:
            return it
        else:
            return (selector(x) for x in it)

    def __aiter__(self):
        selector = self.selector
        ait = self.source.__aiter__()

        if selector is None:
            return ait
        else:
            return (selector(x) async for x in ait)

    def __await__(self):
        return self._result_async().__await__()

    def result(self, timeout=None):
        return list(self)

    async def _result_async(self):
        return [x async for x in self]
