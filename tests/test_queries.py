from functools import wraps
from typing import Iterable, List, Mapping, Tuple

import pytest

from pnq.exceptions import NoElementError, NotFoundError, NotOneElementError
from pnq.queries import DictEx, IndexQuery, ListEx, PairQuery, Query, SetEx, query

# from pnq import query as pnq
pnq = query


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
        assert isinstance(pnq({}), Mapping)
        assert isinstance(pnq(tuple()), Iterable)
        assert isinstance(pnq(set()), Iterable)
        assert isinstance(pnq(frozenset()), Iterable)

    def test_iter_compatibility(self):
        # クエリメソッドで実行する際は、キーバリューを返すように標準化しているが、
        # forのデフォルトの挙動は変えない
        q = pnq({1: "a", 2: "b"})
        assert [x for x in q] == [1, 2]
        assert [x for x in reversed(q)] == [2, 1]

    def test_simple_query(self):
        pnq([1]).map(lambda x: x + 1).to(list) == [2]
        pnq([1]).filter(lambda x: x == 1).to(list) == [1]
        pnq([1]).filter(lambda x: x != 1).to(list) == []


class Test010_Finalizer:
    class TestExecuting:
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

    class TestAggregating:
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
            ],
        )
        def test_concat(self, src, expect):
            assert pnq(src).concat() == expect
            assert pnq(src).map(lambda x: x).concat() == expect

        def test_concat_delimiter(self):
            assert pnq(["a", "b", "c"]).concat(delimiter=",") == "a,b,c"
            assert (
                pnq(["a", "b", "c"]).map(lambda x: x).concat(delimiter=",") == "a,b,c"
            )

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

                assert q.min(selector=lambda x: x[0]) == 0
                assert q.min(selector=lambda x: x[1]) == 10

                assert q.max(selector=lambda x: x[0]) == 0
                assert q.max(selector=lambda x: x[1]) == 10

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

    class TestGetting:
        def test_no_elements(self):
            q = pnq([])

            with pytest.raises(
                TypeError, match="missing 1 required positional argument"
            ):
                q.get()

            with pytest.raises(NotFoundError):
                q.get(1)

            with pytest.raises(NoElementError):
                q.first()

            with pytest.raises(NoElementError):
                q.one()

            with pytest.raises(NoElementError):
                q.last()

            assert q.get(1, None) is None
            assert q.get_or(1, None) is None
            assert q.first_or(None) is None
            assert q.one_or(None) is None
            assert q.last_or(None) is None
            assert q.get(0, 1) == 1
            assert q.get_or(0, 1) == 1
            assert q.first_or(2) == 2
            assert q.one_or(3) == 3

        def test_one_elements(self):
            q = pnq([5])

            with pytest.raises(
                TypeError, match="missing 1 required positional argument"
            ):
                q.get()

            assert q.get(0) == 5
            assert q.first() == 5
            assert q.one() == 5
            assert q.last() == 5

            assert q.get(0, None) == 5
            assert q.get_or(0, None) == 5
            assert q.first_or(None) == 5
            assert q.one_or(None) == 5
            assert q.last_or(None) == 5

            assert q.get(0, 1) == 5
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

            assert q.get(0, None) == -10
            assert q.get_or(0, None) == -10
            assert q.get(1, None) == 10
            assert q.get_or(1, None) == 10
            assert q.first_or(None) == -10

            with pytest.raises(NotOneElementError):
                q.one_or(None)

            assert q.last_or(None) == 10

            assert q.get(0, 1) == -10
            assert q.get_or(0, 1) == -10
            assert q.get(1, 2) == 10
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

        with pytest.raises(TypeError, match="None"):
            pnq([]).map(None)

    def test_map_star(self):
        pass

    def test_map_flat(self):
        pass

    def test_map_recursive(self):
        pass

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

        from pnq.requests import Response

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

            from pnq.requests import Response

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
        assert pnq([1]).filter(lambda x: x == 0).to(list) == []

    def test_filter_type(self):
        assert pnq([1]).filter_type(int).to(list) == [1]
        assert pnq([1]).filter_type(str).to(list) == []
        assert pnq([1]).filter_type(bool).to(list) == []
        assert pnq([True]).filter_type(bool).to(list) == [True]

        # pythonの仕様でboolはintを継承しているのでヒットしてしまう
        assert pnq([True]).filter_type(int).to(list) == [True]

    def test_get_many(self):
        assert pnq([0, 10, 20]).get_many(2).to(list) == [20]
        assert pnq([0, 10, 20]).get_many(0, 1).to(list) == [0, 10]

        db = pnq({1: "a", 2: "b", 3: "c"})
        assert db.get_many(1, 2).to(list) == [(1, "a"), (2, "b")]
        assert db.get_many(4).to(list) == []

        assert pnq((0, 10, 20)).get_many(2).to(list) == [20]
        assert pnq((0, 10, 20)).get_many(1, 0).to(list) == [10, 0]

        assert pnq(set((0, 10, 20))).get_many(20).to(list) == [20]
        assert pnq(set((0, 10, 20))).get_many(10, 20).to(list) == [10, 20]

        assert pnq(frozenset((0, 10, 20))).get_many(20).to(list) == [20]
        assert pnq(frozenset((0, 10, 20))).get_many(10, 20).to(list) == [10, 20]

    def test_unique(self):
        assert pnq([(0, 0), (0, 1), (0, 0)]).unique(lambda x: (x[0], x[1])).to(
            list
        ) == [(0, 0), (0, 1)]

        assert pnq([(0, 0, 0), (0, 1, 1), (0, 0, 2)]).unique(lambda x: (x[0], x[1])).to(
            list
        ) == [(0, 0), (0, 1)]


class Hoge:
    def __init__(self, id, name=""):
        self.id = id
        self.name = name


class Test070_Sort:
    def test_validate(self):
        with pytest.raises(TypeError):
            pnq([]).order_by_attrs()

        pnq([]).order_by_attrs("id")
        pnq([]).order_by_attrs("id", "name")

        with pytest.raises(TypeError):
            pnq([]).order_by_items()

        pnq([]).order_by_items("id")
        pnq([]).order_by_items("id", "name")

    def test_no_elements(self):
        assert pnq([]).reverse().to(list) == []
        assert list(reversed(pnq([]))) == []
        assert pnq({}).reverse().to(list) == []
        assert list(reversed(pnq({}))) == []
        assert pnq([]).order_by_attrs("id").to(list) == []
        assert pnq([]).order_by_items("id").to(list) == []
        assert pnq([]).order(lambda x: x).to(list) == []

    def test_one_elements(self):
        obj = Hoge(id=10)

        assert pnq([1]).reverse().to(list) == [1]
        assert list(reversed(pnq([1]))) == [1]
        assert pnq({1: "a"}).reverse().to(list) == [(1, "a")]
        # pythonの標準動作はreversedはキーのみを返す
        assert list(reversed(pnq({1: "a"}))) == [1]
        assert pnq([obj]).order_by_attrs("id").to(list) == [obj]
        assert pnq([tuple([10])]).order_by_items(0).to(list) == [(10,)]
        assert pnq({1: "a"}).order_by_items(1).to(list) == [(1, "a")]
        assert pnq([obj]).order(lambda x: x.id).to(list) == [obj]

    def test_two_elements(self):
        obj1 = Hoge(id=10, name="b")
        obj2 = Hoge(id=20, name="a")

        assert pnq([2, 1]).reverse().to(list) == [1, 2]
        assert list(reversed(pnq([2, 1]))) == [1, 2]
        assert pnq({2: "a", 1: "a"}).reverse().to(list) == [(1, "a"), (2, "a")]
        # pythonの標準動作はreversedはキーのみを返す
        assert list(reversed(pnq({2: "a", 1: "a"}))) == [1, 2]
        assert pnq([obj2, obj1]).order_by_attrs("id").to(list) == [obj1, obj2]
        assert pnq([obj1, obj2]).order_by_attrs("id").to(list) == [obj1, obj2]
        assert pnq([(2, 0), (1, 100)]).order_by_items(0).to(list) == [(1, 100), (2, 0)]
        assert pnq([(2, 0), (1, 100)]).order_by_items(1).to(list) == [(2, 0), (1, 100)]
        assert pnq({2: "a", 1: "b"}).order_by_items(0).to(list) == [(1, "b"), (2, "a")]
        assert pnq({2: "a", 1: "b"}).order_by_items(1).to(list) == [(2, "a"), (1, "b")]
        assert pnq([obj2, obj1]).order(lambda x: x.id).to(list) == [obj1, obj2]
        assert pnq([obj2, obj1]).order(lambda x: x.name).to(list) == [obj2, obj1]

    def test_multi_value(self):
        obj1 = Hoge(id=10, name="b")
        obj2 = Hoge(id=20, name="a")
        obj3 = Hoge(id=30, name="a")

        assert pnq([obj3, obj2, obj1]).order(lambda x: (x.id, x.name)).to(list) == [
            obj1,
            obj2,
            obj3,
        ]
        assert pnq([obj3, obj2, obj1]).order(lambda x: (x.name, x.id)).to(list) == [
            obj2,
            obj3,
            obj1,
        ]

        assert pnq([obj3, obj2, obj1]).order_by_attrs("id", "name").to(list) == [
            obj1,
            obj2,
            obj3,
        ]
        assert pnq([obj3, obj2, obj1]).order_by_attrs("name", "id").to(list) == [
            obj2,
            obj3,
            obj1,
        ]

        dic1 = dict(id=10, name="b")
        dic2 = dict(id=20, name="a")
        dic3 = dict(id=30, name="a")

        assert pnq([dic3, dic2, dic1]).order_by_items("id", "name").to(list) == [
            dic1,
            dic2,
            dic3,
        ]
        assert pnq([dic3, dic2, dic1]).order_by_items("name", "id").to(list) == [
            dic2,
            dic3,
            dic1,
        ]


class TestSleep:
    def test_sync(self):
        pnq([1, 2, 3]).sleep(0).to(list) == [1, 2, 3]

    def test_async(self):
        import asyncio

        results = []

        async def func():
            async for elm in pnq([1, 2, 3]).sleep_async(0):
                results.append(elm)

        asyncio.run(func())
        assert results == [1, 2, 3]


class TestDict:
    @staticmethod
    def db():
        return pnq({1: "a", 2: "b", 3: "c"})

    def test_init(self):
        from typing import Tuple

        import pnq as pq

        obj1 = pnq({1: "a", 2: "b", 3: "c"})
        obj2 = pnq([(1, "a"), (2, "b"), (3, "c")]).to(dict)
        obj3 = pnq([(1, "a"), (2, "b"), (3, "c")]).to(DictEx[int, str])

        cls = obj1.__class__

        assert isinstance(obj1, cls)
        assert isinstance(obj2, dict)
        assert isinstance(obj3, cls)

    def test_get(self):
        db = self.db()
        assert db[1] == "a"
        assert db.get(1) == "a"
        assert db.get(1, 0) == "a"
        assert db.get_or(1, 10) == "a"
        assert db.get_or(1, None) == "a"
        with pytest.raises(KeyError):
            assert db.get(-1)
        assert db.get(-1, None) is None
        assert db.get(-1, 10) == 10
        assert db.get_or(-1, 10) == 10
        assert db.get_or(-1, None) == None

    def test_get_many(self):
        db = self.db()
        assert db.get_many(1, 2).to(list) == [(1, "a"), (2, "b")]
        assert db.get_many(4).to(list) == []

    def test_other(self):
        db = self.db()
        assert db.keys().to(list) == [1, 2, 3]
        assert db.values().to(list) == ["a", "b", "c"]
        assert db.items().to(list) == [(1, "a"), (2, "b"), (3, "c")]

        assert isinstance(db.to(dict), dict)


class TestSlicer:
    def test_skip(self):
        assert pnq([]).skip(0).to(list) == []
        assert pnq([]).skip(1).to(list) == []
        assert pnq([1]).skip(0).to(list) == [1]
        assert pnq([1]).skip(1).to(list) == []
        assert pnq([1, 2]).skip(0).to(list) == [1, 2]
        assert pnq([1, 2]).skip(1).to(list) == [2]
        assert pnq([1, 2]).skip(2).to(list) == []
        assert pnq([1, 2]).skip(3).to(list) == []

    def test_take(self):
        assert pnq([]).take(0).to(list) == []
        assert pnq([]).take(1).to(list) == []
        assert pnq([1]).take(0).to(list) == []
        assert pnq([1]).take(1).to(list) == [1]
        assert pnq([1, 2]).take(0).to(list) == []
        assert pnq([1, 2]).take(1).to(list) == [1]
        assert pnq([1, 2]).take(2).to(list) == [1, 2]
        assert pnq([1, 2]).take(3).to(list) == [1, 2]

    def test_range(self):
        q = pnq([1, 2, 3, 4, 5, 6])

        assert q.range(0, -1).to(list) == []
        assert q.range(0, 0).to(list) == []

        assert q.range(-1, 0).to(list) == []
        assert q.range(0, 1).to(list) == [1]
        assert q.range(1, 2).to(list) == [2]
        assert q.range(2, 3).to(list) == [3]

        assert q.range(-2, 0).to(list) == []
        assert q.range(0, 2).to(list) == [1, 2]
        assert q.range(2, 4).to(list) == [3, 4]
        assert q.range(4, 6).to(list) == [5, 6]

        assert q.range(5, 6).to(list) == [6]
        assert q.range(5, 7).to(list) == [6]
        assert q.range(6, 7).to(list) == []

    def test_page(self):
        from pnq.queries import page_calc

        with pytest.raises(ValueError):
            page_calc(0, -1)

        assert page_calc(-1, 0) == (0, 0)

        assert page_calc(0, 0) == (0, 0)
        assert page_calc(1, 0) == (0, 0)
        assert page_calc(2, 0) == (0, 0)
        assert page_calc(3, 0) == (0, 0)

        assert page_calc(0, 1) == (-1, 0)
        assert page_calc(1, 1) == (0, 1)
        assert page_calc(2, 1) == (1, 2)
        assert page_calc(3, 1) == (2, 3)

        assert page_calc(0, 2) == (-2, 0)
        assert page_calc(1, 2) == (0, 2)
        assert page_calc(2, 2) == (2, 4)
        assert page_calc(3, 2) == (4, 6)

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

        assert q.page(0, 0).to(list) == []
        assert q.page(1, 0).to(list) == []
        assert q.page(2, 0).to(list) == []
        assert q.page(3, 0).to(list) == []

        assert q.page(0, 1).to(list) == []
        assert q.page(1, 1).to(list) == [1]
        assert q.page(2, 1).to(list) == [2]
        assert q.page(3, 1).to(list) == [3]

        assert q.page(0, 2).to(list) == []
        assert q.page(1, 2).to(list) == [1, 2]
        assert q.page(2, 2).to(list) == [3, 4]
        assert q.page(3, 2).to(list) == [5, 6]
