from .._sync_generate.queries import *  # noqa

_zip = zip
_enumerate = enumerate


def order_by_reverse(source):  # type: ignore
    if hasattr(source, "__reversed__"):
        return reversed(source)
    else:

        def reverse_iterator():
            yield from reversed(list(source))

        return reverse_iterator()
