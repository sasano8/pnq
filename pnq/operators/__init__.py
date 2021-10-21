from typing import Any, Callable

__all__ = ["item", "attr"]


def _get_getter(is_attr):
    if is_attr:
        return getattr
    else:
        return lambda x, k: x[k]


def eq(field, value, attr=False):
    getter = _get_getter(attr)
    return lambda x: getter(x, field) == value


def ne(field, value, attr=False):
    getter = _get_getter(attr)
    return lambda x: getter(x, field) != value


class Item:
    def __getattribute__(self, name: str) -> "ItemOperator":
        return ItemOperator(name)


class Attr:
    def __getattribute__(self, name: str) -> "AttrOperator":
        return AttrOperator(name)


class ItemOperator:
    def __init__(self, key) -> None:
        self.key = key

    def __eq__(self, o: object) -> Callable[[Any], bool]:  # type: ignore
        return eq(self.key, o, attr=False)

    def __ne__(self, o: object) -> Callable[[Any], bool]:  # type: ignore
        return ne(self.key, o, attr=False)


class AttrOperator:
    def __init__(self, key) -> None:
        self.key = key

    def __eq__(self, o: object) -> Callable[[Any], bool]:  # type: ignore
        return eq(self.key, o, attr=True)

    def __ne__(self, o: object) -> Callable[[Any], bool]:  # type: ignore
        return ne(self.key, o, attr=True)


item = Item()
attr = Attr()
