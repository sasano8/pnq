import typing

from pnq import query

DEFAULT_IGNORE_KEYS = {
    "TYPE_CHECKING",
    "runtime_checkable",
    "get_type_hints",
    "get_args",
    "is_typeddict",
    "no_type_check_decorator",
    "final",
    "cast",
    "no_type_check",
    "get_origin",
    "overload",
    # "ForwardRef",
}


def default_ignore_keys():
    return query(DEFAULT_IGNORE_KEYS)


def keys(ignores: typing.Iterable[str] = DEFAULT_IGNORE_KEYS):
    _types = set(typing.__all__)
    for key in ignores:
        _types.discard(key)

    return query(_types)


def values(ignores: typing.Iterable[str] = DEFAULT_IGNORE_KEYS):
    return query(getattr(typing, x) for x in keys(ignores))


def items(ignores: typing.Iterable[str] = DEFAULT_IGNORE_KEYS):
    return query((x, getattr(typing, x)) for x in keys(ignores))
