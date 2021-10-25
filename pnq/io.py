import json
from contextlib import contextmanager

import pnq


@contextmanager
def from_csv(path):
    import csv

    with open(path) as f:
        reader = csv.reader(f)
        yield pnq.query(reader)


@contextmanager
def from_jsonl(path, deserializer=json.loads):
    with open(path) as f:
        strip_lines = (line.strip() for line in f.readlines())
        ignore_empty_lines = (deserializer(x) for x in strip_lines if x)
        yield pnq.query(ignore_empty_lines)
