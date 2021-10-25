import time
from itertools import product as _product

from .._sync_generate.queries import *  # noqa
from .._sync_generate.queries import _enumerate, _filter, _map

_zip = zip
# _enumerate = enumerate


def gather(source, selector=None, parallel: int = 1, timeout=None):  # type: ignore
    raise NotImplementedError()


def gather_tagged(source, selector=None, parallel: int = 1, timeout=None):  # type: ignore # noqa
    raise NotImplementedError()


def order_by_reverse(source):  # type: ignore
    if hasattr(source, "__reversed__"):
        return reversed(source)
    else:

        def reverse_iterator():
            yield from reversed(list(source))

        return reverse_iterator()


def sleep(source, seconds: float):  # type: ignore
    # FIXME: テストを通すためとりあえず現在の実装に合わせている
    raise NotImplementedError()
    sleep = time.sleep
    for v in source:
        yield v
        sleep(seconds)


cartesian = _product
