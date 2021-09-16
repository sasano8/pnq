import inspect
from typing import Iterable, List, Mapping, Set, Tuple, TypeVar, no_type_check

undefined = object()

T = TypeVar("T")


def piter(target):
    if hasattr(target, "__piter__"):
        return target.__piter__()
    elif isinstance(target, Mapping):
        return iter(target.items())
    else:
        return iter(target)


class QueryBase:
    # def __init__(self, func, source: Iterable[T], /, *args, **kwargs) -> None:
    def __init__(self, func, source: Iterable[T], *args, **kwargs) -> None:
        self.func = func
        self.source = source
        self.args = args
        self.kwargs = kwargs

    @no_type_check
    def __iter__(self):
        return self.__piter__()

    def __str__(self) -> str:
        kwargs = [f"{k}={v}" for k, v in self.kwargs.items()]
        params = list(self.args) + kwargs
        params_str = ", ".join(str(self.__inspect_arg(x)) for x in params)
        return f"{self.func.__name__}({params_str})"

    def debug(self):
        info = self.get_upstream(self)
        str_info = [str(x) for x in info]
        info = ".".join(str_info)
        print(info)
        return info

    @classmethod
    def __get_upstream(cls, target):
        if hasattr(target, "source"):
            source = target.source
            return source
        else:
            return None

    @classmethod
    def _get_upstream(cls, target):
        if not target:
            raise TypeError()

        results = [target]

        while target:
            target = cls.__get_upstream(target)
            if target:
                results.append(target)

        return results

    @classmethod
    def get_upstream(cls, target):
        pipe = cls._get_upstream(target)
        pipe.reverse()
        info = [x for x in pipe]
        return info

    @staticmethod
    def __inspect_arg(arg):
        if inspect.isfunction(arg):
            if arg.__name__ == "<lambda>":
                # return inspect.getsourcelines(arg)[0][0]
                sig = inspect.signature(arg)
                return str(sig) + " => ..."
            else:
                return arg.__name__

        elif isinstance(arg, type):
            return arg.__name__
        else:
            return arg

    async def __aiter__(self):
        async for elm in piter(self):
            yield elm

    def is_generator(self):
        return True if self.source is None else False


class LazyGenerator(QueryBase):
    def __init__(self, func, *args, **kwargs):
        self.func = func
        self.source = None
        self.args = args
        self.kwargs = kwargs

        def __piter__(self):
            return self.func(*self.args, **self.kwargs)


class LazyIterate(QueryBase):
    def __piter__(self):
        return self.func(piter(self.source), *self.args, **self.kwargs)


class LazyReference(QueryBase):
    def __piter__(self):
        return self.func(self.source, *self.args, **self.kwargs)


class LazyQuery(QueryBase):
    def __init__(self, source):
        super().__init__(piter, source)


class Argument:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    @staticmethod
    def from_value(value):
        if isinstance(value, dict):
            return Argument(**value)
        elif isinstance(value, tuple):
            return Argument(*value)
        else:
            return Argument(value)

    def push(self, func):
        return func(*self.args, **self.kwargs)
