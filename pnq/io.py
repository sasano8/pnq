import json
from typing import TypeVar

from .base.queries import Query as QueryBase
from .queries import Query

T = TypeVar("T")


class File(Query[T], QueryBase):  # type: ignore
    def __init__(self, *args, **kwargs):
        self.run_iter_type = self.iter_type


class Jsonl(File[dict]):
    def __init__(self, path, deserializer=json.loads):
        super().__init__()
        self.path = path
        self.deserializer = deserializer

    def _impl_iter(self):
        deserializer = self.deserializer
        with open(self.path) as f:
            for line in f.readlines():
                yield deserializer(line)
