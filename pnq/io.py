import json
from contextlib import _GeneratorContextManager
from functools import wraps
from typing import Any, List, cast

from pnq.queries import QueryBase as Query
from pnq.queries import query


class PnqContextManager(_GeneratorContextManager):
    def __iter__(self):
        with self as iter:
            for x in iter:
                yield x


def contextmanager(func):
    @wraps(func)
    def helper(*args, **kwds):
        return PnqContextManager(func, args, kwds)

    return helper


@contextmanager
def from_csv(path, delimiter=","):
    import csv

    with open(path) as f:
        reader = csv.reader(f, delimiter=delimiter)
        yield cast(Query[List[str]], query(reader))


@contextmanager
def from_jsonl(path, deserializer=json.loads):
    with open(path) as f:
        strip_lines = (line.strip() for line in f.readlines())
        ignore_empty_lines = (deserializer(x) for x in strip_lines if x)
        yield cast(Query[Any], query(ignore_empty_lines))


from collections import deque
from inspect import isgeneratorfunction
from io import BytesIO, StringIO
from typing import Iterable, Union


class GeneratorReader:
    def __init__(self, empty: Union[str, bytes], iterable):
        self.cache = BufferCache(empty)
        if isinstance(iterable, Iterable):
            if isinstance(iterable, (str, bytes)):
                iterable = iter([iterable])
            else:
                self.iterable = iter(iterable)
        elif isgeneratorfunction(iterable):
            self.iterable = None
            self.func = iterable
        else:
            raise TypeError()

    def _init(self):
        if self.iterable is None:
            self.iterable = iter(self.func())

    def read(self, size: int = None):
        self._init()
        cache = self.cache
        it = self.iterable

        if size is None:
            for x in it:
                cache.write(x)
            return cache.read()
        else:
            while cache.size < size:
                try:
                    cache.write(next(it))
                except StopIteration:
                    break
            return cache.read(size)


class StrGeneratorReader(GeneratorReader):
    def __init__(self, iterable):
        super().__init__("", iterable)


class BytesGeneratorReader(GeneratorReader):
    def __init__(self, iterable):
        super().__init__(b"", iterable)


class BufferCache:
    def __init__(self, empty: Union[str, bytes] = None):
        self.cache = deque()
        self.size = 0

        if empty is None:
            self._empty = None
        else:
            self._set_empty(empty)

    def __bool__(self):
        return bool(self.size)

    def get_io(self):
        if isinstance(self.empty, str):
            return StringIO()
        elif isinstance(self.empty, bytes):
            return BytesIO()
        else:
            raise TypeError()

    @property
    def empty(self) -> Union[str, bytes]:
        if self._empty is None:
            # 一度もstrかbytesか書き込まれていない場合、何の型の値を返すか判断できないのでエラーとする
            raise RuntimeError()
        else:
            return self._empty

    def _set_empty(self, val):
        if isinstance(val, (str, bytes)):
            self._empty = val.__class__()  # 空文字
        else:
            raise TypeError("Must be str or bytes.")

    def write(self, val):
        if self._empty is None:
            self._set_empty(val)
        self._append(val)

    def read(self, size: int = None):
        limit = size or float("inf")
        buffers = []
        sum = 0

        while self:
            chunk = self._popleft()
            buffers.append(chunk)
            sum += len(chunk)
            if limit <= sum:
                break

        buf_count = len(buffers)

        if buf_count == 0:
            return self.empty

        if buf_count == 1:
            buf = buffers[0]
        else:
            buf = self.empty.join(buffers)

        if size is None:
            return buf
        elif size < len(buf):
            left = buf[:size]
            right = buf[size:]
            self._appendleft(right)
            return left
        else:
            return buf

    def defrag(self, size: int, keep_remain: bool = False):
        if self.size == 0:
            return

        if not keep_remain:
            while chunk := self.read(size):
                yield chunk
            return
        else:
            while size <= self.size:
                buf = self.read(size)
                if buf:
                    yield buf

    def _append(self, val):
        if val:
            self.size += len(val)
            self.cache.append(val)

    def _appendleft(self, val):
        if val:
            self.size += len(val)
            self.cache.appendleft(val)

    def _pop(self):
        val = self.cache.pop()
        self.size -= len(val)
        return val

    def _popleft(self):
        val = self.cache.popleft()
        self.size -= len(val)
        return val
