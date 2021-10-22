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
        return map_iter(self.source, self.selector)

    def __aiter__(self):
        return map_aiter(self.source, self.selector)

    def __await__(self):
        return self._result_async().__await__()

    def result(self, timeout=None):
        return list(self)

    async def _result_async(self):
        return [x async for x in self]


def map_iter(source, selector):
    it = source.__iter__()

    if selector is None:
        return it
    else:
        return (selector(x) for x in it)


def map_aiter(source, selector):
    ait = source.__aiter__()

    if selector is None:
        return ait
    else:
        return (selector(x) async for x in ait)
