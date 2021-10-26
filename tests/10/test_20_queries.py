from functools import partial, wraps
from typing import List

import pytest

from pnq.exceptions import (
    DuplicateElementError,
    MustError,
    MustTypeError,
    NoElementError,
    NotFoundError,
    NotOneElementError,
)
from pnq.queries import query
from tests.conftest import to_sync

pnq = query


class Aiter:
    def __init__(self, source) -> None:
        self.source = source

    async def __aiter__(self):
        for v in self.source:
            yield v


def catch(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        expect = kwargs.get("expect", None)
        if isinstance(expect, Exception):
            msg = str(expect)
            with pytest.raises(expect.__class__, match=msg):
                func(*args, **kwargs)
        else:
            func(*args, **kwargs)

    return wrapper


class Test000_Init:
    def test_create_query(self):
        assert pnq([]).to(list) == []
        assert pnq([]).to(dict) == {}
        assert pnq({}).to(list) == []
        assert pnq({}).to(dict) == {}

        assert pnq([1]).to(list) == [1]
        with pytest.raises(TypeError):
            assert pnq([1]).to(dict) == {}
        assert pnq({1: "a"}).to(list) == [(1, "a")]
        assert pnq({1: "a"}).to(dict) == {1: "a"}
        assert pnq([(1, 2)]).to(list) == [(1, 2)]
        assert pnq([(1, 2)]).to(dict) == {1: 2}

    def test_query_type(self):
        from typing import Iterable, Mapping

        assert isinstance(pnq([]), Iterable)
        assert not isinstance(pnq([]), Mapping)
        assert not isinstance(pnq({}), Mapping)  # クエリ化したら辞書互換でないように変更
        assert isinstance(pnq({}), Iterable)
        assert isinstance(pnq(tuple()), Iterable)
        assert isinstance(pnq(set()), Iterable)
        assert isinstance(pnq(frozenset()), Iterable)

    def test_simple_query(self):
        pnq([1]).map(lambda x: x + 1).to(list) == [2]
        pnq([1]).filter(lambda x: x == 1).to(list) == [1]
        pnq([1]).filter(lambda x: x != 1).to(list) == []

    @pytest.mark.parametrize(
        "src",
        [[1], {1: "a"}, {1}, frozenset([1]), (1, 2), (x for x in range(1)), Aiter([1])],
    )
    def test_forbid_protocol(self, src):
        # 関連するプロトコルを実装すると副作用が生じるので実装しない
        # __iter__と__aiter__のみ可能なことを強調する

        # __len__を実装していると、list(obj)とした時に無限ループが起きる原因になる
        query = pnq(src)
        assert not hasattr(query, "__len__")
        assert not hasattr(query.map(lambda x: x), "__len__")

        # 副作用が多いので実装しない
        assert not hasattr(query, "__getitem__")
        assert not hasattr(query.map(lambda x: x), "__getitem__")

        # keysを実装しているとdict(obj)時にdict型と勘違いされるため実装しない
        assert not hasattr(query, "keys")
        assert not hasattr(query.map(lambda x: x), "keys")

        # lenとgetitemを実装しなければ副作用はなさそうだが実装しないこととする
        assert not hasattr(query, "__reversed__")
        assert not hasattr(query.map(lambda x: x), "__reversed__")

        from typing import Iterable, Mapping

        assert isinstance(query, Iterable)
        assert not isinstance(query, Mapping)  # マッピングでないこと

        try:
            assert list(query)
            if isinstance(src, dict):
                assert dict(query)

        except Exception as e:
            msg = str(e)
            if not "can't __iter__" in msg:
                raise

    @pytest.mark.parametrize(
        "src, expect",
        [
            ([1, 2], [2, 1]),
            ({1: "a", 2: "b"}, [(2, "b"), (1, "a")]),
            ({1}, [1]),
            (frozenset([1]), [1]),
            ((1, 2), [2, 1]),
            ((x for x in range(1)), [1]),
        ],
    )
    def test_compatibility(self, src, expect):
        with pytest.raises(TypeError, match="object is not reversible"):
            reversed(pnq(src))

        with pytest.raises(TypeError, match="has no len"):
            len(pnq(src))

        with pytest.raises(TypeError, match="object is not subscriptable"):
            pnq(src)[0]

    def test_save(self):
        from pnq import list as plist

        assert pnq([]).save() == []
        assert isinstance(pnq([]).save(), plist)
        assert pnq([1, 2, 3]).save() == [1, 2, 3]
        assert pnq((1, 2, 3)).save() == [1, 2, 3]
        assert pnq(tuple([1, 2, 3])).save() == [1, 2, 3]
        assert pnq({"a": 1}).save() == [("a", 1)]
        result = pnq({1, 2, 3}).save()
        assert len(result) == 3
        assert 1 in result
        assert 2 in result
        assert 3 in result

        async def aiter():
            yield 1
            yield 2
            yield 3

        async def main():
            result = await pnq(aiter())
            assert result == [1, 2, 3]
            assert isinstance(result, plist)

        import asyncio

        asyncio.run(main())


class Test009_Sleep:
    def test_sync(self):
        pnq([1, 2, 3]).sleep(0).to(list) == [1, 2, 3]

    def test_async(self):
        import asyncio

        results = []

        async def func():
            async for elm in pnq([1, 2, 3]).sleep(0):
                results.append(elm)

        asyncio.run(func())
        assert results == [1, 2, 3]


class Test010_Finalizer:
    @pytest.mark.parametrize(
        "src, typ, expect",
        [
            ([], list, []),
            ([], dict, {}),
            ([], tuple, tuple()),
            ([], set, set()),
            ([], frozenset, frozenset()),
            ([1], list, [1]),
            ([1], dict, TypeError("cannot convert dictionary")),
            ([1], tuple, tuple([1])),
            ([1], set, set([1])),
            ([1], frozenset, frozenset([1])),
            ([(1, 2)], list, [(1, 2)]),
            ([(1, 2)], dict, {1: 2}),
            ([(1, 2)], tuple, ((1, 2),)),
            ([(1, 2)], set, set([(1, 2)])),
            ([(1, 2)], frozenset, frozenset([(1, 2)])),
            ({1: 2}, list, [(1, 2)]),
            ({1: 2}, dict, {1: 2}),
            ({1: 2}, tuple, ((1, 2),)),
            ({1: 2}, set, set([(1, 2)])),
            ({1: 2}, frozenset, frozenset([(1, 2)])),
        ],
    )
    @catch
    def test_to_from_type(self, src, typ, expect):
        obj = pnq(src).to(typ)
        assert obj == expect
        assert isinstance(obj, typ)

        obj = pnq(src).map(lambda x: x).to(typ)
        assert obj == expect
        assert isinstance(obj, typ)

    def test_to_from_func(self):
        def to_list(iterator):
            return list(iterator)

        assert pnq([]).to(to_list) == []
        assert pnq([1]).to(to_list) == [1]
        assert pnq({1: 2}).to(to_list) == [(1, 2)]

    @pytest.mark.skipif(True, reason="[each_asyncは不要だと思う。除去予定]")
    def test_each(self):
        import asyncio

        assert pnq([]).each() is None
        assert pnq([]).each_unpack() is None
        assert asyncio.run(pnq([]).each_async()) is None
        assert asyncio.run(pnq([]).each_async_unpack()) is None

        result_each = []  # type: ignore
        result_each_unpack = []

        def add_result_each_unpack(**kwargs):
            result_each_unpack.append(kwargs)

        async def async_add_result_each(x):
            result_each.append(x)

        async def async_add_result_each_unpack(**kwargs):
            result_each_unpack.append(kwargs)

        pnq([1]).each(result_each.append)
        assert result_each == [1]

        pnq([{"name": "test", "age": 20}]).each_unpack(add_result_each_unpack)
        assert result_each_unpack == [{"name": "test", "age": 20}]

        asyncio.run(pnq([2]).each_async(async_add_result_each))
        assert result_each == [1, 2]

        asyncio.run(
            pnq([{"name": "test2", "age": 50}]).each_async_unpack(
                async_add_result_each_unpack
            )
        )
        assert result_each_unpack == [
            {"name": "test", "age": 20},
            {"name": "test2", "age": 50},
        ]

    @pytest.mark.parametrize(
        "src, expect",
        [
            ([], 0),
            ({}, 0),
            (tuple(), 0),
            (set(), 0),
            (frozenset(), 0),
            ([1], 1),
            ({"a": 1}, 1),
            (tuple([1]), 1),
            (set([1]), 1),
            (frozenset([1]), 1),
        ],
    )
    def test_len(self, src, expect):
        assert pnq(src).len() == expect
        assert pnq(src).map(lambda x: x).len() == expect

    @pytest.mark.parametrize(
        "src, expect",
        [
            ([], False),
            ({}, False),
            (tuple(), False),
            (set(), False),
            (frozenset(), False),
            ([1], True),
            ({"a": 1}, True),
            (tuple([1]), True),
            (set([1]), True),
            (frozenset([1]), True),
        ],
    )
    def test_exists(self, src, expect):
        assert pnq(src).exists() == expect
        assert pnq(src).map(lambda x: x).exists() == expect

    @pytest.mark.parametrize(
        "src, expect",
        [
            ([], True),
            ({}, True),
            (tuple(), True),
            (set(), True),
            (frozenset(), True),
            ([1], True),
            ({"a": 1}, True),
            (tuple([1]), True),
            (set([1]), True),
            (frozenset([1]), True),
            ([0], False),
            ([0, 1], False),
            ([1, 1], True),
        ],
    )
    def test_all(self, src, expect):
        assert pnq(src).all() == expect
        assert pnq(src).map(lambda x: x).all() == expect

    @pytest.mark.parametrize(
        "src, expect",
        [
            ([], False),
            ({}, False),
            (tuple(), False),
            (set(), False),
            (frozenset(), False),
            ([1], True),
            ({"a": 1}, True),
            (tuple([1]), True),
            (set([1]), True),
            (frozenset([1]), True),
            ([0], False),
            ([0, 1], True),
        ],
    )
    def test_any(self, src, expect):
        assert pnq(src).any() == expect
        assert pnq(src).map(lambda x: x).any() == expect

    @pytest.mark.parametrize(
        "src, value, expect",
        [
            ([], 1, False),
            ({}, 1, False),
            (tuple(), 1, False),
            (set(), 1, False),
            (frozenset(), 1, False),
            ([1], 1, True),
            ({1: "a"}, 1, False),
            ({1: "a"}, (1, "a"), True),
            (tuple([1]), 1, True),
            (set([1]), 1, True),
            (frozenset([1]), 1, True),
        ],
    )
    def test_contains(self, src, value, expect):
        assert pnq(src).contains(value) == expect
        assert pnq(src).map(lambda x: x).contains(value) == expect

    @pytest.mark.parametrize(
        "src, expect",
        [
            ([], ValueError("arg is an empty sequence")),
            ({}, ValueError("arg is an empty sequence")),
            (tuple(), ValueError("arg is an empty sequence")),
            (set(), ValueError("arg is an empty sequence")),
            (frozenset(), ValueError("arg is an empty sequence")),
            ([1], 1),
            ({"a": 1}, ("a", 1)),
            (tuple([1]), 1),
            (set([1]), 1),
            (frozenset([1]), 1),
            ([1, 2], 1),
            ([2, 1, 2], 1),
        ],
    )
    @catch
    def test_min(self, src, expect):
        assert pnq(src).min() == expect
        assert pnq(src).map(lambda x: x).min() == expect

    def test_min_default(self):
        assert pnq([]).min(default=-1) == -1
        assert pnq([100]).min(default=-1) == 100  # 要素が存在する場合は要素が優先

    @pytest.mark.parametrize(
        "src, expect",
        [
            ([], ValueError("arg is an empty sequence")),
            ({}, ValueError("arg is an empty sequence")),
            (tuple(), ValueError("arg is an empty sequence")),
            (set(), ValueError("arg is an empty sequence")),
            (frozenset(), ValueError("arg is an empty sequence")),
            ([1], 1),
            ({"a": 1}, ("a", 1)),
            (tuple([1]), 1),
            (set([1]), 1),
            (frozenset([1]), 1),
            ([1, 2], 2),
            ([1, 2, 1], 2),
        ],
    )
    @catch
    def test_max(self, src, expect):
        assert pnq(src).max() == expect
        assert pnq(src).map(lambda x: x).max() == expect

    def test_max_default(self):
        assert pnq([]).max(default=-1) == -1
        assert pnq([1]).max(default=10) == 1  # 要素が存在する場合は要素が優先

    @pytest.mark.parametrize(
        "src, expect",
        [
            ([], 0),
            ({}, 0),
            (tuple(), 0),
            (set(), 0),
            (frozenset(), 0),
            ([1], 1),
            ({"a": 1}, TypeError("unsupported")),
            (tuple([1]), 1),
            (set([1]), 1),
            (frozenset([1]), 1),
        ],
    )
    @catch
    def test_sum(self, src, expect):
        assert pnq(src).sum() == expect
        assert pnq(src).map(lambda x: x).sum() == expect

    @pytest.mark.parametrize(
        "src, expect",
        [
            ([], 0),
            ({}, 0),
            (tuple(), 0),
            (set(), 0),
            (frozenset(), 0),
            ([1], 1),
            ({"a": 1}, Exception("not a number")),
            (tuple([1]), 1),
            (set([1]), 1),
            (frozenset([1]), 1),
            ([1, 2], 1.5),
        ],
    )
    @catch
    def test_average(self, src, expect):
        assert pnq(src).average() == expect
        assert pnq(src).map(lambda x: x).average() == expect

    @pytest.mark.parametrize(
        "src, seed, op, expect",
        [
            ([], 0, "+=", 0),
            ({}, 0, "+=", 0),
            (tuple(), 0, "+=", 0),
            (set(), 0, "+=", 0),
            (frozenset(), 0, "+=", 0),
            ([1], 0, "+=", 1),
            ({"a": 1}, 0, "+=", TypeError("unsupported operand type")),
            (tuple([1]), 0, "+=", 1),
            (set([1]), 0, "+=", 1),
            (frozenset([1]), 0, "+=", 1),
        ],
    )
    @catch
    def test_reduce(self, src, seed, op, expect):
        assert pnq(src).reduce(seed, op) == expect
        assert pnq(src).map(lambda x: x).reduce(seed, op) == expect

    @pytest.mark.parametrize(
        "src, expect",
        [
            ([], ""),
            ({}, ""),
            (tuple(), ""),
            (set(), ""),
            (frozenset(), ""),
            ([1], "1"),
            ({"a": 1}, "('a', 1)"),
            (tuple([1]), "1"),
            (set([1]), "1"),
            (frozenset([1]), "1"),
            (["a", "b", "c"], "abc"),
            ([None], ""),
        ],
    )
    def test_concat(self, src, expect):
        assert pnq(src).concat() == expect
        assert pnq(src).map(lambda x: x).concat() == expect

    def test_concat_delimiter(self):
        assert pnq(["a", "b", "c"]).concat(delimiter=",") == "a,b,c"
        assert pnq(["a", "b", "c"]).map(lambda x: x).concat(delimiter=",") == "a,b,c"

    def test_selectors(self):
        no_accept = "unexpected keyword argument 'selector'"

        def test_selector_sub(q):
            with pytest.raises(TypeError, match=no_accept):
                q.len(selector=lambda x: x)

            with pytest.raises(TypeError, match=no_accept):
                q.exists(selector=lambda x: x)

            assert q.all(selector=lambda x: x[0]) == False
            assert q.all(selector=lambda x: x[1])

            assert q.any(selector=lambda x: x[0]) == False
            assert q.any(selector=lambda x: x[1])

            assert q.contains(10, selector=lambda x: x[0]) == False
            assert q.contains(10, selector=lambda x: x[1])

            assert q.min(key_selector=lambda x: x[0]) == (0, 10)
            assert q.min(key_selector=lambda x: x[1]) == (0, 10)

            assert q.max(key_selector=lambda x: x[0]) == (0, 10)
            assert q.max(key_selector=lambda x: x[1]) == (0, 10)

            assert q.sum(selector=lambda x: x[0]) == 0
            assert q.sum(selector=lambda x: x[1]) == 10

            assert q.average(selector=lambda x: x[0]) == 0
            assert q.average(selector=lambda x: x[1]) == 10

            assert q.reduce(0, "+=", selector=lambda x: x[0]) == 0
            assert q.reduce(0, "+=", selector=lambda x: x[1]) == 10

            assert q.concat(selector=lambda x: x[0]) == "0"
            assert q.concat(selector=lambda x: x[1]) == "10"

        test_selector_sub(pnq([(0, 10)]))
        test_selector_sub(pnq([(0, 10)]).map(lambda x: x))

    def test_no_elements(self):
        q = pnq([])

        with pytest.raises(TypeError, match="missing 1 required positional argument"):
            q.get()

        with pytest.raises(NotFoundError):
            q.get(1)

        with pytest.raises(NoElementError):
            q.first()

        with pytest.raises(NoElementError):
            q.one()

        with pytest.raises(NoElementError):
            q.last()

        assert q.get_or(1, None) is None
        assert q.first_or(None) is None
        assert q.one_or(None) is None
        assert q.last_or(None) is None
        assert q.get_or(0, 1) == 1
        assert q.first_or(2) == 2
        assert q.one_or(3) == 3

    def test_one_elements(self):
        q = pnq([5])

        with pytest.raises(TypeError, match="missing 1 required positional argument"):
            q.get()

        assert q.get(0) == 5
        assert q.first() == 5
        assert q.one() == 5
        assert q.last() == 5

        assert q.get_or(0, None) == 5
        assert q.first_or(None) == 5
        assert q.one_or(None) == 5
        assert q.last_or(None) == 5

        assert q.get_or(0, 1) == 5
        assert q.first_or(2) == 5
        assert q.one_or(3) == 5
        assert q.last_or(4) == 5

    def test_two_elements(self):
        q = pnq([-10, 10])

        assert q.get(0) == -10
        assert q.get(1) == 10
        assert q.first() == -10
        with pytest.raises(NotOneElementError):
            q.one()
        assert q.last() == 10

        assert q.get_or(0, None) == -10
        assert q.get_or(1, None) == 10
        assert q.first_or(None) == -10

        with pytest.raises(NotOneElementError):
            q.one_or(None)

        assert q.last_or(None) == 10

        assert q.get_or(0, 1) == -10
        assert q.get_or(1, 2) == 10
        assert q.first_or(3) == -10
        assert q.first_or(3) == -10

        with pytest.raises(NotOneElementError):
            q.one_or(4)

        assert q.last_or(5) == 10

    def test_get_or_raise(self):
        q = pnq([10])
        assert q.get_or_raise(0, "err") == 10
        assert q.one_or_raise("err") == 10
        assert q.first_or_raise("err") == 10
        assert q.last_or_raise("err") == 10

        q = pnq([])
        with pytest.raises(Exception, match="err"):
            q.get_or_raise(0, "err") == 10
        with pytest.raises(Exception, match="err"):
            q.one_or_raise("err") == 10
        with pytest.raises(Exception, match="err"):
            q.first_or_raise("err") == 10
        with pytest.raises(Exception, match="err"):
            q.last_or_raise("err") == 10

        with pytest.raises(NotFoundError, match="err"):
            q.get_or_raise(0, NotFoundError("err")) == 10
        with pytest.raises(NotFoundError, match="err"):
            q.one_or_raise(NotFoundError("err")) == 10
        with pytest.raises(NotFoundError, match="err"):
            q.first_or_raise(NotFoundError("err")) == 10
        with pytest.raises(NotFoundError, match="err"):
            q.last_or_raise(NotFoundError("err")) == 10

        q = pnq([10, 20])
        assert q.get_or_raise(1, "err") == 20
        with pytest.raises(NotOneElementError):
            q.one_or_raise("err")
        assert q.first_or_raise("err") == 10
        assert q.last_or_raise("err") == 20


class Test020_Transform:
    def test_map(self):
        assert pnq([1]).map(lambda x: x * 2).to(list) == [2]
        assert pnq([None]).map(str).to(list) == [""]
        assert str(None) == "None"

        with pytest.raises(TypeError, match="missing"):
            pnq([]).map()

        assert pnq([]).map(None).to(list) == []

    def test_unpack_pos(self):
        assert pnq([(1, 2)]).unpack_pos(lambda k, v: k).to(list) == [1]

    def test_unpack_kw(self):
        assert pnq([{"name": "test", "age": 20}]).unpack_kw(lambda name, age: name).to(
            list
        ) == ["test"]

    def test_unpack(self):
        # mapはアンパックできないこと
        with pytest.raises(
            TypeError, match="missing 1 required positional argument: 'v'"
        ):
            pnq([(1, 2)]).map(lambda k, v: k).to(list)

    def test_select(self):
        assert pnq([{"name": "a"}]).select("name").to(list) == ["a"]
        assert pnq([str]).select("__name__", attr=True).to(list) == ["str"]
        assert pnq([dict(id=1, name="a")]).select("id", "name").to(list) == [(1, "a")]
        assert pnq([str]).select("__name__", "__class__", attr=True).to(list) == [
            ("str", type)
        ]

        with pytest.raises(TypeError, match="itemgetter expected 1"):
            pnq([]).select()

        with pytest.raises(TypeError, match="attrgetter expected 1"):
            pnq([]).select(attr=True)

    def test_select_as_tuple(self):
        assert pnq([1]).select_as_tuple().to(list) == [()]
        assert pnq([(10, 20)]).select_as_tuple(0).to(list) == [(10,)]
        assert pnq([(10, 20)]).select_as_tuple(1).to(list) == [(20,)]
        assert pnq([(10, 20)]).select_as_tuple(0, 1).to(list) == [(10, 20)]
        assert pnq([(10, 20)]).select_as_tuple(1, 0).to(list) == [(20, 10)]

        assert pnq([1]).select_as_tuple(attr=True).to(list) == [()]
        assert pnq([str]).select_as_tuple("__name__", attr=True).to(list) == [("str",)]
        assert pnq([str]).select_as_tuple("__class__", attr=True).to(list) == [(type,)]
        assert pnq([str]).select_as_tuple("__name__", "__class__", attr=True).to(
            list
        ) == [("str", type)]
        assert pnq([str]).select_as_tuple("__class__", "__name__", attr=True).to(
            list
        ) == [(type, "str")]

    def test_select_as_dict(self):
        assert pnq([1]).select_as_dict().to(list) == [{}]
        assert pnq([(10, 20)]).select_as_dict(0).to(list) == [{0: 10}]
        assert pnq([(10, 20)]).select_as_dict(1).to(list) == [{1: 20}]
        assert pnq([(10, 20)]).select_as_dict(0, 1).to(list) == [{0: 10, 1: 20}]
        assert pnq([(10, 20)]).select_as_dict(1, 0).to(list) == [{1: 20, 0: 10}]

        assert pnq([1]).select_as_dict(attr=True).to(list) == [{}]
        assert pnq([str]).select_as_dict("__name__", attr=True).to(list) == [
            {"__name__": "str"}
        ]
        assert pnq([str]).select_as_dict("__class__", attr=True).to(list) == [
            {"__class__": type}
        ]
        assert pnq([str]).select_as_dict("__name__", "__class__", attr=True).to(
            list
        ) == [{"__name__": "str", "__class__": type}]
        assert pnq([str]).select_as_dict("__class__", "__name__", attr=True).to(
            list
        ) == [{"__class__": type, "__name__": "str"}]

        assert pnq([str]).select_as_dict("__xxx__", attr=True, default=100).to(
            list
        ) == [{"__xxx__": 100}]

        assert pnq([str]).select_as_dict(
            "__name__", "__xxx__", attr=True, default=100
        ).to(list) == [{"__name__": "str", "__xxx__": 100}]

    def test_reflect(self):
        # transposed = transpose(
        #     {
        #         "id": ["id"],
        #         "name": {"name", "searchable"},
        #         "kana": {"kana", "searchable"},
        #         "note": ["searchable"],
        #     }
        # )

        # single, multi = split_single_multi(transposed)

        # assert single == {
        #     "id": "id",
        #     "name": "name",
        #     "kana": "kana",
        # }

        # assert multi == {
        #     "searchable": ["name", "kana", "note"],
        # }
        result = (
            pnq([{"id": 1, "name": "山田", "kana": "yamada", "note": "hoge"}])
            .reflect(
                {
                    "id": "id",
                    "name": {"name", "searchable"},
                    "kana": {"kana", "searchable"},
                    "note": "searchable",
                }
            )
            .to(list)
        )

        assert result[0] == {
            "id": 1,
            "name": "山田",
            "kana": "yamada",
            "searchable": ["山田", "yamada", "hoge"],
        }

    def test_map_star(self):
        pass

    def test_flat(self):
        assert pnq([]).flat().to(list) == []

        with pytest.raises(TypeError, match="object is not iterable"):
            pnq([1]).flat().to(list)

        assert pnq(["abc"]).flat().to(list) == ["a", "b", "c"]
        assert pnq([(0, "abc")]).flat(lambda x: x[1]).to(list) == ["a", "b", "c"]
        assert pnq([(0, [1, 2, 3])]).flat(lambda x: x[1]).to(list) == [1, 2, 3]

    def test_flat_recursive(self):
        data = [
            {
                "name": "root1",
                "nodes": [
                    {
                        "name": "n1",
                        "nodes": [
                            {"name": "n2", "nodes": []},
                            {"name": "n3", "nodes": [{"name": "n4", "nodes": []}]},
                        ],
                    }
                ],
            },
            {
                "name": "root2",
                "nodes": [{"name": "n5", "nodes": []}],
            },
        ]
        assert pnq(data).flat_recursive(lambda x: x["nodes"]).select("name").to(
            list
        ) == ["root1", "n1", "n2", "n3", "n4", "root2", "n5"]

    def test_cast(self):
        # castは型注釈を誤魔化す
        # 内部的に何もしないため、同じクエリオブジェクトを参照していることを検証する
        q = pnq([1])
        assert q.cast(str) == q

    def test_enumerate(self):
        assert pnq([1]).enumerate().to(list) == [(0, 1)]

    def test_group_by(self):
        data = [
            {"name": "banana", "color": "yellow", "count": 3},
            {"name": "apple", "color": "red", "count": 2},
            {"name": "strawberry", "color": "red", "count": 5},
        ]

        # カラー別名前
        pnq(data).group_by(lambda x: (x["color"], x["name"])).to(list) == [
            ("yellow", ["banana"]),
            ("red", ["apple", "strawberry"]),
        ]

        # カラー別個数
        pnq(data).select("color", "count").group_by().to(list) == [
            ("yellow", [3]),
            ("red", [2, 5]),
        ]

    def test_chunked(self):
        assert pnq([]).chunked(1).to(list) == []
        assert pnq([1, 2, 3]).chunked(1).to(list) == [[1], [2], [3]]
        assert pnq([1, 2, 3, 4, 5]).chunked(2).to(list) == [[1, 2], [3, 4], [5]]

    def test_tee(self):
        pass

    def test_join(self):
        pass

    def test_group_join(self):
        pass

    def test_pivot_unstack(self):
        data = [
            {"name": "test1", "age": 20},
            {"name": "test2", "age": 25},
            {"name": "test3", "age": 30, "sex": "male"},
        ]
        result1 = pnq(data).pivot_unstack().to(dict)

        assert result1 == {
            "name": ["test1", "test2", "test3"],
            "age": [20, 25, 30],
            "sex": [None, None, "male"],
        }

        result2 = pnq(data).pivot_unstack(default="").to(dict)

        assert result2 == {
            "name": ["test1", "test2", "test3"],
            "age": [20, 25, 30],
            "sex": ["", "", "male"],
        }

    def test_pivot_stack(self):
        data = {
            "name": ["test1", "test2", "test3"],
            "age": [20, 25, 30],
            "sex": [None, None, "male"],
        }

        result = pnq(data).pivot_stack().to(list)
        assert result == [
            {"name": "test1", "age": 20, "sex": None},
            {"name": "test2", "age": 25, "sex": None},
            {"name": "test3", "age": 30, "sex": "male"},
        ]

    def test_pivot_stack_unstack(self):
        data = [
            {"name": "test1", "age": 20, "sex": ""},
            {"name": "test2", "age": 25, "sex": ""},
            {"name": "test3", "age": 30, "sex": "male"},
        ]
        result = pnq(data).pivot_unstack().pivot_stack().to(list)
        assert result == data

        data = {
            "name": ["test1", "test2", "test3"],
            "age": [20, 25, 30],
            "sex": [None, None, "male"],
        }
        result = pnq(data).pivot_stack().pivot_unstack().to(dict)
        assert result == data

    def test_debug(self):
        result = []

        assert pnq([1]).debug(lambda x: x).to(list) == [1]
        assert pnq([1]).debug(result.append).to(list) == [1]
        assert result == [1]

        assert pnq([2]).debug(lambda x: x, printer=result.append).to(list) == [2]
        assert result == [1, 2]

        q = pnq([3])

        @q.debug
        def break_point(x):
            result.append(x)

        assert break_point.to(list) == [3]
        assert result == [1, 2, 3]

    def test_request(self):
        from datetime import datetime

        # from pnq.base.requests import Response
        from pnq._itertools.requests import Response

        result = []

        def ok(value):
            result.append(value)
            return "ok"

        def err(value1, value2):
            raise Exception("error")

        response: List[Response] = pnq([{"value": 1}]).request(ok).to(list)
        assert result == [1]
        res = response[0]
        assert res.func == ok
        assert res.kwargs == {"value": 1}
        assert res.err is None
        assert res.result == "ok"
        assert isinstance(res.start, datetime)
        assert isinstance(res.end, datetime)

        response: List[Response] = (
            pnq([{"value1": 1, "value2": 2}]).request(err).to(list)
        )
        assert result == [1]
        res = response[0]
        assert res.func == err
        assert res.kwargs == {"value1": 1, "value2": 2}
        assert str(res.err) == "error"
        assert res.result is None
        assert isinstance(res.start, datetime)
        assert isinstance(res.end, datetime)

    def test_request_async(self):
        import asyncio

        async def main():
            from datetime import datetime

            # from pnq.base.requests import Response
            from pnq._itertools.requests import Response

            result = []

            async def ok(value):
                result.append(value)
                return "ok"

            async def err(value1, value2):
                raise Exception("error")

            q = pnq([{"value": 1}]).request_async(ok)
            response: List[Response] = [x async for x in q]

            assert result == [1]
            res = response[0]
            assert res.func == ok
            assert res.kwargs == {"value": 1}
            assert res.err is None
            assert res.result == "ok"
            assert isinstance(res.start, datetime)
            assert isinstance(res.end, datetime)

            q = pnq([{"value1": 1, "value2": 2}]).request_async(err)
            response: List[Response] = [x async for x in q]

            assert result == [1]
            res = response[0]
            assert res.func == err
            assert res.kwargs == {"value1": 1, "value2": 2}
            assert str(res.err) == "error"
            assert res.result is None
            assert isinstance(res.start, datetime)
            assert isinstance(res.end, datetime)

        asyncio.run(main())


class Test030_Filter:
    def test_filter(self):
        assert pnq([1]).filter(lambda x: x == 1).to(list) == [1]
        assert pnq([1]).filter(lambda x: x == 0).to(list) == []

        with pytest.raises(TypeError, match="missing"):
            pnq([1]).filter().to(list)

    def test_filter_type(self):
        assert pnq([1]).filter_type(int).to(list) == [1]
        assert pnq([1]).filter_type(str).to(list) == []
        assert pnq([1]).filter_type(bool).to(list) == []
        assert pnq([True]).filter_type(bool).to(list) == [True]

        # pythonの仕様でboolはintを継承しているのでヒットしてしまう
        assert pnq([True]).filter_type(int).to(list) == [True]

    def test_filter_keys(self):
        assert pnq([0, 10, 20]).filter_keys(2).to(list) == [20]
        assert pnq([0, 10, 20]).filter_keys(0, 1).to(list) == [0, 10]

        db = pnq({1: "a", 2: "b", 3: "c"})
        assert db.filter_keys(1, 2).to(list) == [(1, "a"), (2, "b")]
        assert db.filter_keys(4).to(list) == []

        assert pnq((0, 10, 20)).filter_keys(2).to(list) == [20]
        assert pnq((0, 10, 20)).filter_keys(1, 0).to(list) == [10, 0]

        assert pnq(set((0, 10, 20))).filter_keys(20).to(list) == [20]
        assert pnq(set((0, 10, 20))).filter_keys(10, 20).to(list) == [10, 20]

        assert pnq(frozenset((0, 10, 20))).filter_keys(20).to(list) == [20]
        assert pnq(frozenset((0, 10, 20))).filter_keys(10, 20).to(list) == [10, 20]

    def test_unique(self):
        assert pnq([(0, 0), (0, 1), (0, 0)]).filter_unique(lambda x: (x[0], x[1])).to(
            list
        ) == [(0, 0), (0, 1)]

        assert pnq([(0, 0, 0), (0, 1, 1), (0, 0, 2)]).filter_unique(
            lambda x: (x[0], x[1])
        ).to(list) == [(0, 0), (0, 1)]


class Test040_Must:
    def test_must(self):
        with pytest.raises(TypeError, match="missing"):
            pnq([1]).must().to(list)

        assert pnq([1]).must(lambda x: x == 1).to(list) == [1]

        with pytest.raises(MustError):
            assert pnq([1]).must(lambda x: x == 0).to(list) == []

        with pytest.raises(MustError, match="12345"):
            pnq([1]).must(lambda x: x == 0, msg="12345").to(list)

    def test_must_type(self):
        with pytest.raises(TypeError, match="missing"):
            pnq([1]).must_type().to(list)

        with pytest.raises(MustTypeError, match="is not .*str"):
            pnq([1]).must_type(str).to(list)

        assert pnq([1]).must_type(int).to(list)

        with pytest.raises(MustTypeError, match="is not .*list.*dict"):
            pnq([1]).must_type(list, dict).to(list)

        assert pnq([1]).must_type(list, int).to(list)
        assert pnq([1]).must_type(int, list).to(list)

    def test_must_unique(self):
        assert pnq([]).must_unique().to(list) == []
        assert pnq([1]).must_unique().to(list) == [1]

        with pytest.raises(DuplicateElementError, match="1"):
            assert pnq([1, 1]).must_unique().to(list) == [1]

        with pytest.raises(DuplicateElementError, match="1"):
            assert pnq([(0, 1), (0, 1)]).must_unique().to(list) == [1]

        assert pnq([(0, 1, 1), (0, 1, 2)]).must_unique().to(list) == [
            (0, 1, 1),
            (0, 1, 2),
        ]

        with pytest.raises(DuplicateElementError):
            pnq([(0, 1, 1), (0, 1, 2)]).must_unique(lambda x: (x[0], x[1])).to(list)

    def test_must_keys(self):
        # list
        assert pnq([]).must_keys().to(list) == []

        with pytest.raises(NotFoundError, match="0"):
            pnq([]).must_keys(0).to(list)

        assert pnq([10]).must_keys(0).to(list) == [10]

        with pytest.raises(NotFoundError, match="1"):
            pnq([10]).must_keys(1).to(list)

        assert pnq([10, 20]).must_keys(0, 1).to(list) == [10, 20]

        with pytest.raises(NotFoundError, match="2"):
            pnq([10, 20]).must_keys(1, 2).to(list)

        # dict
        assert pnq({"a": 1, "b": 2, "c": 3}).must_keys("b", "c").to(list) == [
            ("b", 2),
            ("c", 3),
        ]

        with pytest.raises(NotFoundError, match="d"):
            pnq({"a": 1, "b": 2, "c": 3}).must_keys("d").to(list)

        # tuple
        assert pnq((1, 2, 3)).must_keys(1, 2).to(list) == [
            2,
            3,
        ]

        with pytest.raises(NotFoundError, match="3"):
            pnq((1, 2, 3)).must_keys(3).to(list)

        # set
        assert pnq(set((1, 2, 3))).must_keys(2, 3).to(list) == [
            2,
            3,
        ]

        with pytest.raises(NotFoundError, match="4"):
            pnq(set((1, 2, 3))).must_keys(4).to(list)

        # frozen set
        assert pnq(frozenset((1, 2, 3))).must_keys(2, 3).to(list) == [
            2,
            3,
        ]

        with pytest.raises(NotFoundError, match="4"):
            pnq(frozenset((1, 2, 3))).must_keys(4).to(list)


class Test050_Partition:
    def test_take(self):
        assert pnq([]).take(0).to(list) == []
        assert pnq([]).take(1).to(list) == []
        assert pnq([1]).take(0).to(list) == []
        assert pnq([1]).take(1).to(list) == [1]
        assert pnq([1, 2]).take(0).to(list) == []
        assert pnq([1, 2]).take(1).to(list) == [1]
        assert pnq([1, 2]).take(2).to(list) == [1, 2]
        assert pnq([1, 2]).take(3).to(list) == [1, 2]

    def test_take_range(self):
        q = pnq([1, 2, 3, 4, 5, 6])

        assert q.take(range(0, -1)).to(list) == []
        assert q.take(range(0, 0)).to(list) == []

        assert q.take(range(-1, 0)).to(list) == []
        assert q.take(range(0, 1)).to(list) == [1]
        assert q.take(range(1, 2)).to(list) == [2]
        assert q.take(range(2, 3)).to(list) == [3]

        assert q.take(range(-2, 0)).to(list) == []
        assert q.take(range(0, 2)).to(list) == [1, 2]
        assert q.take(range(2, 4)).to(list) == [3, 4]
        assert q.take(range(4, 6)).to(list) == [5, 6]

        assert q.take(range(5, 6)).to(list) == [6]
        assert q.take(range(5, 7)).to(list) == [6]
        assert q.take(range(6, 7)).to(list) == []

    def test_take_while(self):
        pass

    def test_skip(self):
        assert pnq([]).skip(0).to(list) == []
        assert pnq([]).skip(1).to(list) == []
        assert pnq([1]).skip(0).to(list) == [1]
        assert pnq([1]).skip(1).to(list) == []
        assert pnq([1, 2]).skip(0).to(list) == [1, 2]
        assert pnq([1, 2]).skip(1).to(list) == [2]
        assert pnq([1, 2]).skip(2).to(list) == []
        assert pnq([1, 2]).skip(3).to(list) == []

    def test_skip_range(self):
        pass

    def test_skip_while(self):
        pass

    def test_take_page(self):
        # from pnq.actions import take_page_calc
        from pnq._itertools._sync.queries import _take_page_calc as take_page_calc

        with pytest.raises(ValueError):
            take_page_calc(1, -1)

        take_page_calc(1, 0)

        with pytest.raises(ValueError):
            take_page_calc(0, 0)

        # assert take_page_calc(-1, 0) == range(0, 0)

        # assert take_page_calc(0, 0) == range(0, 0)
        assert take_page_calc(1, 0) == range(0, 0)
        assert take_page_calc(2, 0) == range(0, 0)
        assert take_page_calc(3, 0) == range(0, 0)

        # assert take_page_calc(0, 1) == range(-1, 0)
        assert take_page_calc(1, 1) == range(0, 1)
        assert take_page_calc(2, 1) == range(1, 2)
        assert take_page_calc(3, 1) == range(2, 3)

        # assert take_page_calc(0, 2) == range(-2, 0)
        assert take_page_calc(1, 2) == range(0, 2)
        assert take_page_calc(2, 2) == range(2, 4)
        assert take_page_calc(3, 2) == range(4, 6)

        arr = [1, 2, 3, 4, 5, 6]

        assert arr[0:0] == []
        assert arr[0:0] == []
        assert arr[0:0] == []
        assert arr[0:0] == []

        assert arr[-1:0] == []
        assert arr[0:1] == [1]
        assert arr[1:2] == [2]
        assert arr[2:3] == [3]

        assert arr[-2:0] == []
        assert arr[0:2] == [1, 2]
        assert arr[2:4] == [3, 4]
        assert arr[4:6] == [5, 6]

        q = pnq(arr)

        assert q.take_page(0, 0).to(list) == []
        assert q.take_page(1, 0).to(list) == []
        assert q.take_page(2, 0).to(list) == []
        assert q.take_page(3, 0).to(list) == []

        assert q.take_page(0, 1).to(list) == []
        assert q.take_page(1, 1).to(list) == [1]
        assert q.take_page(2, 1).to(list) == [2]
        assert q.take_page(3, 1).to(list) == [3]

        assert q.take_page(0, 2).to(list) == []
        assert q.take_page(1, 2).to(list) == [1, 2]
        assert q.take_page(2, 2).to(list) == [3, 4]
        assert q.take_page(3, 2).to(list) == [5, 6]


class Test060_Sort:
    def test_order_by_map(self):
        assert pnq([]).order_by_map().to(list) == []
        assert pnq([]).order_by_map(lambda x: x).to(list) == []
        assert pnq([]).order_by_map(None).to(list) == []

        assert pnq([1]).order_by_map().to(list) == [1]
        assert pnq([1]).order_by_map(lambda x: x).to(list) == [1]
        assert pnq([1]).order_by_map(None).to(list) == [1]

        assert pnq([2, 1]).order_by_map().to(list) == [1, 2]
        assert pnq([2, 1]).order_by_map(lambda x: x).to(list) == [1, 2]
        assert pnq([2, 1]).order_by_map(desc=True).to(list) == [2, 1]

        assert pnq([2, 1]).order_by_map(desc=True).to(list) == [2, 1]
        assert pnq([2, 1]).order_by_map(lambda x: x, desc=True).to(list) == [2, 1]

        assert pnq([(2, 1), (1, 2)]).order_by_map(lambda x: x[0]).to(list) == [
            (1, 2),
            (2, 1),
        ]

        assert pnq([(2, 1), (1, 2)]).order_by_map(lambda x: x[0], desc=True).to(
            list
        ) == [
            (2, 1),
            (1, 2),
        ]

        assert pnq([(2, 1), (1, 2)]).order_by_map(lambda x: x[1], desc=True).to(
            list
        ) == [
            (1, 2),
            (2, 1),
        ]

        assert pnq([(0, 1, 2), (0, 2, 1), (0, 1, 1)]).order_by_map(
            lambda x: (x[1], x[2])
        ).to(list) == [(0, 1, 1), (0, 1, 2), (0, 2, 1)]

        assert pnq([(0, 1, 2), (0, 2, 1), (0, 1, 1)]).order_by_map(
            lambda x: (x[2], x[1])
        ).to(list) == [(0, 1, 1), (0, 2, 1), (0, 1, 2)]

    def test_order_by(self):
        assert pnq([]).order_by().to(list) == []
        assert pnq([]).order_by(attr=True).to(list) == []
        assert pnq([]).order_by(attr=False).to(list) == []
        assert pnq([]).order_by(desc=False).to(list) == []
        assert pnq([]).order_by(desc=True).to(list) == []

        assert pnq([1]).order_by().to(list) == [1]
        assert pnq([1]).order_by(attr=True).to(list) == [1]
        assert pnq([1]).order_by(attr=False).to(list) == [1]
        assert pnq([1]).order_by(desc=False).to(list) == [1]
        assert pnq([1]).order_by(desc=True).to(list) == [1]
        assert pnq([(1, 2)]).order_by(0).to(list) == [(1, 2)]
        assert pnq([(1, 2)]).order_by(1).to(list) == [(1, 2)]

        with pytest.raises(IndexError, match="out of range"):
            assert pnq([(1, 2)]).order_by(2).to(list) == [(1, 2)]

        user_1_b = dict(id=1, name="b")
        user_2_a = dict(id=2, name="a")

        with pytest.raises(TypeError, match="not supported between instances"):
            assert pnq([user_1_b, user_2_a]).order_by().to(list) == [user_1_b, user_2_a]

        assert pnq([user_1_b, user_2_a]).order_by("id").to(list) == [user_1_b, user_2_a]
        assert pnq([user_1_b, user_2_a]).order_by("id", desc=True).to(list) == [
            user_2_a,
            user_1_b,
        ]

        assert pnq([user_1_b, user_2_a]).order_by("name").to(list) == [
            user_2_a,
            user_1_b,
        ]

        assert pnq([user_1_b, user_2_a]).order_by("name", desc=True).to(list) == [
            user_1_b,
            user_2_a,
        ]

        class User:
            def __init__(self, id, name):
                self.id = id
                self.name = name

        user_1_b = User(**user_1_b)  # type: ignore
        user_2_a = User(**user_2_a)  # type: ignore

        with pytest.raises(TypeError, match="not supported between instances"):
            assert pnq([user_1_b, user_2_a]).order_by().to(list) == [user_1_b, user_2_a]

        assert pnq([user_1_b, user_2_a]).order_by("id", attr=True).to(list) == [
            user_1_b,
            user_2_a,
        ]
        assert pnq([user_1_b, user_2_a]).order_by("id", attr=True, desc=True).to(
            list
        ) == [
            user_2_a,
            user_1_b,
        ]

        assert pnq([user_1_b, user_2_a]).order_by("name", attr=True).to(list) == [
            user_2_a,
            user_1_b,
        ]

        assert pnq([user_1_b, user_2_a]).order_by("name", attr=True, desc=True).to(
            list
        ) == [
            user_1_b,
            user_2_a,
        ]

    def test_order_by_reverse(self):
        assert pnq([]).order_by_reverse().to(list) == []
        assert pnq({}).order_by_reverse().to(list) == []

        assert pnq([1]).order_by_reverse().to(list) == [1]
        assert pnq({1: "a"}).order_by_reverse().to(list) == [(1, "a")]

        assert pnq([2, 1]).order_by_reverse().to(list) == [1, 2]
        assert list(reversed([2, 1])) == [1, 2]
        assert pnq({2: "a", 1: "a"}).order_by_reverse().to(list) == [(1, "a"), (2, "a")]
        assert list(reversed([(2, "a"), (1, "a")])) == [(1, "a"), (2, "a")]

        assert pnq(([2, 1])).order_by_reverse().to(list) == [1, 2]

        # セットは順序を保持しないので要素が含まれているかのみ確認
        q = pnq(set([2, 1])).order_by_reverse().to(list)
        assert len(q) == 2
        for v in [1, 2]:
            assert v in q

        # セットは順序を保持しないので要素が含まれているかのみ確認
        q = pnq(frozenset([2, 1])).order_by_reverse().to(list)
        assert len(q) == 2
        for v in [1, 2]:
            assert v in q

        assert pnq((x for x in range(3))).order_by_reverse().to(list) == [2, 1, 0]

    def test_order_by_shuffle(self):
        base = [1, 2, 3]
        results = []

        for i in range(100):
            result = pnq([1, 2, 3]).order_by_shuffle().to(list)
            results.append(result)

        different = 0

        for result in results:
            if result != base:
                different += 1

        # 100 - (100 / 6) = 84
        assert different > (84 * 0.5)  # 50%まで偏りを許容する


class Test070_Scaling:
    def test_cartesian(self):
        assert pnq([1, 2, 3]).cartesian().to(list) == [(1,), (2,), (3,)]
        assert pnq([1, 2, 3]).cartesian([4, 5, 6]).to(list) == [
            (1, 4),
            (1, 5),
            (1, 6),
            (2, 4),
            (2, 5),
            (2, 6),
            (3, 4),
            (3, 5),
            (3, 6),
        ]
        assert pnq([1, 2]).cartesian([3, 4], [5, 6]).to(list) == [
            (1, 3, 5),
            (1, 3, 6),
            (1, 4, 5),
            (1, 4, 6),
            (2, 3, 5),
            (2, 3, 6),
            (2, 4, 5),
            (2, 4, 6),
        ]


class Test500_Type:
    def test_dict_get(self):
        db = pnq({1: "a", 2: "b", 3: "c"})
        assert db.get(1) == "a"
        # assert db.get(1, 0) == "a"
        assert db.get_or(1, 10) == "a"
        assert db.get_or(1, None) == "a"
        with pytest.raises(NotFoundError):
            assert db.get(-1)
        # assert db.get(-1, None) is None
        # assert db.get(-1, 10) == 10
        assert db.get_or(-1, 10) == 10
        assert db.get_or(-1, None) is None

    def test_dict_filter_keys(self):
        db = pnq({1: "a", 2: "b", 3: "c"})
        assert db.filter_keys(1, 2).to(list) == [(1, "a"), (2, "b")]
        assert db.filter_keys(4).to(list) == []
        assert isinstance(db.to(dict), dict)


def add_one(x: int):
    return x + 1


async def add_two(x: int):
    return x + 2


def add_val(v):
    return v


async def add_val_async(v):
    return v


async def queue_to_list(queue):
    import asyncio
    import queue as Queue
    from concurrent.futures import ThreadPoolExecutor

    loop = asyncio.get_running_loop()

    result = []
    while True:
        try:
            result = await loop.run_in_executor(ThreadPoolExecutor(1), queue.get)
            result.append(result)
        except Queue.Empty:
            break
    return result


class Test600_Concurrent:
    @to_sync
    async def test_parallel_processpool(self):
        from pnq.concurrent import ProcessPool as Pool

        with Pool(1) as pool:
            assert (
                pnq([1])
                .parallel(add_one, executor=pool)
                .parallel(add_two, executor=pool)
                .result()
            ) == [4]

            assert (
                await pnq([1])
                .parallel(add_one, executor=pool)
                .parallel(add_two, executor=pool)
            ) == [4]

    @to_sync
    async def test_parallel_threadpool(self):
        from pnq.concurrent import ThreadPool as Pool

        with Pool(1) as pool:
            assert (
                pnq([1])
                .parallel(add_one, executor=pool)
                .parallel(add_two, executor=pool)
                .result()
            ) == [4]

            assert (
                await pnq([1])
                .parallel(add_one, executor=pool)
                .parallel(add_two, executor=pool)
            ) == [4]

    @to_sync
    async def test_parallel_asyncpool(self):
        from pnq.concurrent import AsyncPool as Pool

        async with Pool(1) as pool:
            # assert (
            #     pnq([1])
            #     .parallel(add_one, executor=pool)
            #     .parallel(add_two, executor=pool)
            #     .result()
            # ) == [4]

            assert (
                await pnq([1])
                .parallel(add_one, executor=pool)
                .parallel(add_two, executor=pool)
            ) == [4]

    @to_sync
    async def test_parallel_dummypool(self):
        from pnq.concurrent import DummyPool as Pool

        with Pool(1) as pool:
            assert (
                pnq([1]).parallel(add_one, executor=pool)
                # .parallel(add_two, executor=pool)
                .result()
            ) == [2]

            assert (
                await pnq([1])
                .parallel(add_one, executor=pool)
                .parallel(add_two, executor=pool)
            ) == [4]

    @to_sync
    async def test_parallel_dummypool_default(self):
        assert (
            pnq([1]).parallel(add_one)
            # .parallel(add_two, executor=pool)
            .result()
        ) == [2]

        assert (await pnq([1]).parallel(add_one).parallel(add_two)) == [4]

    @to_sync
    async def test_dispatch_processpool(self):
        from pnq.concurrent import ProcessPool as Pool

        result = []

        def callback(future):
            print(future)
            result.append(future.result())
            print(result)

        async with Pool(1) as pool:
            pnq([1]).dispatch(add_val, executor=pool, callback=callback)
            await pnq([2])._.dispatch(add_val, executor=pool, callback=callback)
            await pnq([3])._.dispatch(add_val_async, executor=pool, callback=callback)

        assert result == [1, 2, 3]

    @to_sync
    async def test_dispatch_threadpool(self):
        from pnq.concurrent import ThreadPool as Pool

        with Pool(1) as pool:
            assert (
                pnq([1])
                .parallel(add_one, executor=pool)
                .parallel(add_two, executor=pool)
                .result()
            ) == [4]

            assert (
                await pnq([1])
                .parallel(add_one, executor=pool)
                .parallel(add_two, executor=pool)
            ) == [4]

    @to_sync
    async def test_dispatch_asyncpool(self):
        from pnq.concurrent import AsyncPool as Pool

        async with Pool(1) as pool:
            # assert (
            #     pnq([1])
            #     .parallel(add_one, executor=pool)
            #     .parallel(add_two, executor=pool)
            #     .result()
            # ) == [4]

            assert (
                await pnq([1])
                .parallel(add_one, executor=pool)
                .parallel(add_two, executor=pool)
            ) == [4]

    @to_sync
    async def test_dispatch_dummypool(self):
        from pnq.concurrent import DummyPool as Pool

        with Pool(1) as pool:
            assert (
                pnq([1]).parallel(add_one, executor=pool)
                # .parallel(add_two, executor=pool)
                .result()
            ) == [2]

            assert (
                await pnq([1])
                .parallel(add_one, executor=pool)
                .parallel(add_two, executor=pool)
            ) == [4]

    @to_sync
    async def test_dispatch_dummypool_default(self):
        assert (
            pnq([1]).parallel(add_one)
            # .parallel(add_two, executor=pool)
            .result()
        ) == [2]

        assert (await pnq([1]).parallel(add_one).parallel(add_two)) == [4]
