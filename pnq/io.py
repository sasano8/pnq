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
