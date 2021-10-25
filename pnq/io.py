import json
from typing import Any, TypeVar

# from .base.queries import Query as QueryBase
from pnq.types import Arguments

from ._itertools.querables import Query as QueryBase
from .queries import Query

T = TypeVar("T")


class File(Query[T], QueryBase):  # type: ignore
    def __init__(self):
        self.run_iter_type = self.iter_type

    def _impl_iter(self):
        x = self._args
        return self._sit(*x.args, **x.kwargs)

    def _impl_aiter(self):
        x = self._args
        return self._ait(*x.args, **x.kwargs)


class Jsonl(File[Any]):
    def __init__(self, path, deserializer=json.loads):
        super().__init__()
        self._args = Arguments(path, deserializer)

    def _sit(self, path, deserializer):
        with open(path) as f:
            for line in [x.strip() for x in f.readlines()]:
                if line:
                    yield deserializer(line)

    async def _ait(self, path, deserializer):
        with open(path) as f:
            for line in [x.strip() for x in f.readlines()]:
                if line:
                    yield deserializer(line)
