import time
from itertools import product as _product

from .._sync_generate.queries import *  # noqa
from .._sync_generate.queries import _enumerate, _filter, _map, _take_page_calc

_zip = zip
# _enumerate = enumerate


def order_by_reverse(source):  # type: ignore
    if hasattr(source, "__reversed__"):
        return reversed(source)
    else:

        def reverse_iterator():
            yield from reversed(list(source))

        return reverse_iterator()


def sleep(source, seconds: float):  # type: ignore
    sleep = time.sleep
    for v in source:
        yield v
        sleep(seconds)


cartesian = _product


def chain(*iterables):  # type: ignore
    for iterable in iterables:
        for x in iterable:
            yield x
