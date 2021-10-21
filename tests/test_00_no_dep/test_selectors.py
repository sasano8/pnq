import pytest

import pnq
from pnq import selectors


class Args:
    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs


class Obj:
    def __init__(self, **kwargs) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)


def return_args(*args, **kwargs):
    return args, kwargs


def test_map():
    assert selectors.map(return_args, unpack="")(1, a="a") == ((1,), {"a": "a"})
    assert selectors.map(return_args, unpack="*")([1, 2]) == ((1, 2), {})
    assert selectors.map(return_args, unpack="**")({"a": 1, "b": 2}) == (
        tuple(),
        {"a": 1, "b": 2},
    )
    assert selectors.map(return_args, unpack="***")(Args(1, 2, a=3, b=4)) == (
        (1, 2),
        {"a": 3, "b": 4},
    )


def test_select():
    dic = {"a": 1, "b": 2}
    obj = Obj(a=1, b=2)

    with pytest.raises(Exception):
        selectors.select()

    with pytest.raises(Exception):
        selectors.select(attr=True)

    with pytest.raises(Exception):
        selectors.select(attr=False)

    assert selectors.select("a")(dic) == 1
    assert selectors.select("a", attr=False)(dic) == 1
    assert selectors.select("a", attr=True)(obj) == 1
    assert selectors.select("a", "b")(dic) == (1, 2)
    assert selectors.select("a", "b", attr=False)(dic) == (1, 2)
    assert selectors.select("a", "b", attr=True)(obj) == (1, 2)


def test_select_as_tuple():
    dic = {"a": 1, "b": 2}
    obj = Obj(a=1, b=2)
    assert selectors.select_as_tuple()(dic) == tuple()
    assert selectors.select_as_tuple(attr=False)(dic) == tuple()
    assert selectors.select_as_tuple(attr=True)(obj) == tuple()
    assert selectors.select_as_tuple("a")(dic) == (1,)
    assert selectors.select_as_tuple("a", attr=False)(dic) == (1,)
    assert selectors.select_as_tuple("a", attr=True)(obj) == (1,)
    assert selectors.select_as_tuple("a", "b")(dic) == (1, 2)
    assert selectors.select_as_tuple("a", "b", attr=False)(dic) == (1, 2)
    assert selectors.select_as_tuple("a", "b", attr=True)(obj) == (1, 2)


def test_select_as_dict():
    dic = {"a": 1, "b": 2}
    obj = Obj(a=1, b=2)
    assert selectors.select_as_dict()(dic) == {}
    assert selectors.select_as_dict(attr=False)(dic) == {}
    assert selectors.select_as_dict(attr=True)(obj) == {}
    assert selectors.select_as_dict("a")(dic) == {"a": 1}
    assert selectors.select_as_dict("a", attr=False)(dic) == {"a": 1}
    assert selectors.select_as_dict("a", attr=True)(obj) == {"a": 1}
    assert selectors.select_as_dict("a", "b")(dic) == {"a": 1, "b": 2}
    assert selectors.select_as_dict("a", "b", attr=False)(dic) == {"a": 1, "b": 2}
    assert selectors.select_as_dict("a", "b", attr=True)(obj) == {"a": 1, "b": 2}


def test_reflect():
    item_getter = pnq.selectors.reflect(
        {
            "id": "id",
            "name": {"name", "searchable"},
            "kana": {"kana", "searchable"},
            "note": "searchable",
        }
    )
    attr_getter = pnq.selectors.reflect(
        {
            "id": "id",
            "name": {"name", "searchable"},
            "kana": {"kana", "searchable"},
            "note": "searchable",
        },
        attr=True,
    )

    assert item_getter({"id": 1, "name": "山田", "kana": "yamada", "note": "hoge"}) == {
        "id": 1,
        "name": "山田",
        "kana": "yamada",
        "searchable": ["山田", "yamada", "hoge"],
    }

    assert attr_getter(Obj(id=1, name="山田", kana="yamada", note="hoge")) == {
        "id": 1,
        "name": "山田",
        "kana": "yamada",
        "searchable": ["山田", "yamada", "hoge"],
    }
