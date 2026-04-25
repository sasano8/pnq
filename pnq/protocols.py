import asyncio
import concurrent
from typing import Dict, Iterable, Protocol, runtime_checkable

"""
concurrent.futures.ProcessPoolExecutorはmultiprocessing.Poolのラッパーです。
https://stackoverflow.com/questions/38311431/concurrent-futures-processpoolexecutor-vs-multiprocessing-pool-pool
"""


@runtime_checkable
class Executor(Protocol):
    def submit(self, *args, **kwargs) -> concurrent.futures.Future:
        ...


class PExecutor(Protocol):
    def submit(self, func, *args, **kwargs) -> concurrent.futures.Future:
        ...

    def asubmit(self, func, *args, **kwargs) -> asyncio.Future:
        ...

    @property
    def running_task_count(self) -> int:
        ...

    @property
    def is_full(self) -> bool:
        ...

    @property
    def is_closed(self) -> bool:
        ...

    @property
    def max_workers(self) -> int:
        ...

    @property
    def is_cpubound(self) -> bool:
        ...


class PExecutable(Protocol):
    def __executor__(self) -> PExecutor:
        ...


import keyword

def format_call(cls_name: str, args: tuple, kwargs: dict) -> str:
    parts = [repr(arg) for arg in args]

    for key, value in kwargs.items():
        if isinstance(key, str) and key.isidentifier() and not keyword.iskeyword(key):
            parts.append(f"{key}={value!r}")
        else:
            # key が a-b や class などの場合は **{...} に逃がす
            parts.append(f"**{{{key!r}: {value!r}}}")

    return f"{cls_name}({', '.join(parts)})"


from typing import Any, Callable, NamedTuple, Union

class WrapPlaceholder:
    def __repr__(self):
        return "*"
    
    def __str__(self):
        return self.__repr__()

class WrapFrame(NamedTuple):
    factory: Union[Callable[..., Any], None]
    target: Any
    args: tuple = ()
    kwargs: dict[str, Any] = {}

    def __get_wrapframe__(self) -> "WrapFrame":
        return self

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return format_call(
            self.factory.__name__,
            (self.target, *self.args),
            self.kwargs,
        )

    def replace(self, target):
        return self._replace(target=target)

    def wrap(self):
        if self.factory is None:
            raise Exception("Cannot wrap an empty Wrappable")

        return self.factory(
            self.target,
            *self.args,
            **self.kwargs,
        )
    
    def is_term(self):
        return not isinstance(self.target, (Wrapped, WrapFrame))
    
    def to_frames_src(self):
        frames, src = unwrap_recursive(self)
        return frames, src


class WrapFrameChain(list):
    def wrap(self, root):
        if not self:
            raise Exception("Chain is empty")

        prev = root
        for x in self:
            frame = x.replace(prev)
            prev = frame.wrap()
        return prev


class Wrapped:
    def __init__(self, _, /, *args, **kwargs):
        self._target = _
        self._args = args
        self._kwargs = kwargs

    def __get_wrapframe__(self) -> WrapFrame:
        return WrapFrame(self.__class__, self._target, self._args, self._kwargs)

    def __str__(self):
        return self.__get_wrapframe__().__str__()

    def __repr__(self):
        return self.__get_wrapframe__().__repr__()


def _unwrap_recursive(target):
    if isinstance(target, (Wrapped, WrapFrame)):
        frame = target.__get_wrapframe__()
    else:
        yield target
        return

    if frame.is_term():
        yield frame.replace(WrapPlaceholder())
        yield frame.target
    else:
        yield frame.replace(WrapPlaceholder())
        yield from _unwrap_recursive(frame.target)

def unwrap_recursive(target):
    elms = list(_unwrap_recursive(target))
    src = elms[-1]
    frames = elms[:-1]
    frames.reverse()
    return WrapFrameChain(frames), src

"""
from pnq import protocols
w = protocols.Wrapped(1,2,3,a=1,b=2)
w2 = protocols.Wrapped(w)
w3 = protocols.Wrapped(w2)
print(w)
repr(w)
print(w3)
repr(w3)
frames, src = w3.__get_wrapframe__().to_frames_src()
frames
src
repr(frames.wrap(src))
"""
