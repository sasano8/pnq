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

from pathlib import Path
import os
from typing import TypedDict


class LocalFileReader:
    """File reader for local filesystem."""

    def __init__(self, file):
        self._file = file

    async def read(self, size: int = -1) -> bytes:
        return self._file.read(size)
    
    async def close(self):
        self._file.close()
    
    async def __aenter__(self):
        return self
        
    async def __aexit__(self, exc_type, exc, tb):
        await self.close()


class LocalFileWriter:
    """File writer for local filesystem."""

    def __init__(self, file):
        self._file = file

    async def write(self, data: bytes) -> int:
        return self._file.write(data)

    async def close(self):
        self._file.close()

    async def __aenter__(self):
        return self
        
    async def __aexit__(self, exc_type, exc, tb):
        await self.close()


class FileInfo(TypedDict):
    """File information."""

    key: str
    size: int
    updated_at: float | None


class FileStore:
    @classmethod
    async def create(cls, root: str = None, create_if_not_exists = True) -> "FileStore":
        if root is None:
            _root = Path(os.getcwd())
        else:
            _root = Path(root)

        if create_if_not_exists:
            _root.parent.mkdir(parents=True, exist_ok=True)

        obj =  cls(_root)
        return obj

    def __init__(self, root: Path):
        self._root = Path(root)

    async def is_connected(self) -> bool:
        return True
    
    async def open_reader(self, key: str):
        if not await self.exists(key):
            raise Exception(f"Key {key} not exists")

        f = (self._root / key).open("rb")
        return LocalFileReader(f)

    async def open_writer(self, key: str):
        (self._root / key).parent.mkdir(parents=True, exist_ok=True)
        f = (self._root / key).open("wb")
        return LocalFileWriter(f)

    async def delete(self, key: str) -> int:
        exists = await self.exists(key)
        (self._root / key).unlink(missing_ok=True)
        return int(exists)
    
    async def stat(self, key: str) -> FileInfo | None:
        if not (self._root / key).is_file():
            return None
        
        stat = (self._root / key).stat()
        return FileInfo(key=key, size=stat.st_size, updated_at=stat.st_mtime)

    async def exists(self, key: str) -> bool:
        stat = await self.stat(key)
        return stat is not None

    async def list(self, limit: int = 10) -> list:
        ...

class DefaultAdapter:
    @classmethod
    def create(cls, read_type, write_type):
        async def load(f):
            value = await f.read()
            return value
        
        async def dump(value, f):
            await f.write(value)

        return load, dump

class JsonAdapter:
    @classmethod
    def create(cls, read_type, write_type):
        import json

        async def load(f):
            value = await f.read()
            decoded = json.loads(value)
            return decoded
        
        async def dump(value, f):
            encoded = json.dumps(value).encode("utf-8")
            await f.write(encoded)

        return load, dump


class KVStore:
    def __init__(self, store: FileStore, adapter_factory = DefaultAdapter.create):
        self._store = store
        self.set_adapter(adapter_factory)

    def set_adapter(self, adapter_factory):
        loader, dumper = adapter_factory(bytes, bytes)
        self.load = loader
        self.dump = dumper

    async def load(self, f):
        value = await f.read()
        return value

    async def dump(self, value: bytes, f):
        await f.write(value)
    
    async def create(self, key: str, value: bytes):
        exists = await self._store.exists(key)
        if exists:
            raise Exception(f"Key {key} already exists")
        
        return await self.put(key, value)

    async def put(self, key: str, value: bytes):
        async with self._store.open_writer(key) as f:
            await self.dump(value, f)

    async def delete(self, key: str):
        return await self._store.delete(key)

    async def list(self, limit: int = 10):
        return await self._store.list(limit)

    async def exists(self, key: str):
        return await self._store.exists(key)

    async def get(self, key: str):
        undefined = object()
        result = await self.get_or_default(key, undefined)
        if result is undefined:
            raise Exception(f"Key {key} not exists")
        
        return result

    async def get_or_default(self, key: str, default = None):
        exists = await self._store.exists(key)
        if not exists:
            return default

        async with self._store.open_reader(key) as f:
            return await self.load(f)
