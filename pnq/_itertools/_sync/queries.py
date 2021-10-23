from .._sync_generate.queries import *  # noqa

_zip = zip
_enumerate = enumerate


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
