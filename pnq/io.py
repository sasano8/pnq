import json
from contextlib import contextmanager
from typing import Any, List, cast

import pnq
from pnq.queries import QueryBase as Query


@contextmanager
def from_csv(path):
    import csv

    with open(path) as f:
        reader = csv.reader(f)
        yield cast(Query[List[str]], pnq.query(reader))


@contextmanager
def from_jsonl(path, deserializer=json.loads):
    with open(path) as f:
        strip_lines = (line.strip() for line in f.readlines())
        ignore_empty_lines = (deserializer(x) for x in strip_lines if x)
        yield cast(Query[Any], pnq.query(ignore_empty_lines))
