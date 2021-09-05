import pytest

from pnq import pnq
from pnq.exceptions import NoElementError, NotOneError


def block(func):
    func()
    return func


def test_type():
    from typing import Iterable, Mapping

    q1 = pnq([])
    q2 = pnq({})

    assert isinstance(q1, Iterable)

    # 辞書のインターフェースを実装したいが、辞書にすると初期化時に誤動作するのでMappingと認識されないようにする
    assert not isinstance(q2, Mapping)


def test_init():
    assert pnq([]).to_list() == []
    assert pnq([]).to_dict() == {}
    assert pnq({}).to_list() == []
    assert pnq({}).to_dict() == {}

    assert pnq([1]).to_list() == [1]
    with pytest.raises(TypeError):
        assert pnq([1]).to_dict() == {}
    assert pnq({1: "a"}).to_list() == [(1, "a")]
    assert pnq({1: "a"}).to_dict() == {1: "a"}
    assert pnq([(1, 2)]).to_list() == [(1, 2)]
    assert pnq([(1, 2)]).to_dict() == {1: 2}


def test_easy():

    # assert pnq([{"key": 3, "name": "test"}]).lookup(lambda x: x["key"]).to_list() == [
    #     (3, {"key": 3, "name": "test"})
    # ]
    # obj = pnq([{"key": 3, "name": "test"}]).lookup(lambda x: x["key"]).to_dict()
    # assert obj[3] == {"key": 3, "name": "test"}
    pass


class TestFilter:
    def test_filter(self):
        assert pnq([1]).filter(lambda x: x == 1).to_list() == [1]
        assert pnq([1]).filter(lambda x: x == 0).to_list() == []
        assert pnq([1]).filter(lambda x: x == 0).to_list() == []

    def test_filter_type(self):
        assert pnq([1]).filter_type(int).to_list() == [1]
        assert pnq([1]).filter_type(str).to_list() == []
        assert pnq([1]).filter_type(bool).to_list() == []
        assert pnq([True]).filter_type(bool).to_list() == [True]
        assert pnq([True]).filter_type(int).to_list() == [
            True
        ]  # pythonの仕様でboolはintを継承しているヒットしてしまう


class TestMap:
    def test_map(self):
        assert pnq([1]).map(lambda x: x * 2).to_list() == [2]
        assert pnq([None]).map(str).to_list() == [""]
        assert str(None) == "None"

    def test_select(self):
        assert pnq([{"name": "a"}]).select("name").to_list() == ["a"]
        assert pnq([str]).select_attr("__name__").to_list() == ["str"]
        # assert pnq([dict(id=1)]).select_items("id").to_list() == [(1,)]
        assert pnq([dict(id=1, name="a")]).select_items("id", "name").to_list() == [
            (1, "a")
        ]
        assert pnq([dict(id=1, name="a", age=5)]).select_items(
            "id", "name"
        ).to_list() == [(1, "a")]
        # assert pnq([str]).select_attrs("__name__").to_list() == [("str",)]
        assert pnq([str]).select_attrs("__name__", "__class__").to_list() == [
            ("str", type)
        ]

    def test_unpack(self):
        with pytest.raises(
            TypeError, match="missing 1 required positional argument: 'v'"
        ):
            pnq([(1, 2)]).map(lambda k, v: k).to_list()
        assert pnq([(1, 2)]).unpack(lambda k, v: k).to_list() == [1]
        assert pnq([{"name": "test", "age": 20}]).unpack_kw(
            lambda name, age: name
        ).to_list() == ["test"]

    def test_enumrate(self):
        assert pnq([1]).enumerate().to_list() == [(0, 1)]

    def test_cast(self):
        # castは型注釈を誤魔化す
        # 内部的に何もしないため、同じクエリオブジェクトを参照していることを検証する
        q = pnq([1])
        assert q.cast(str) == q


def test_aggregator():
    assert pnq([]).len() == 0
    assert pnq({}).len() == 0
    assert pnq([1]).len() == 1
    assert pnq({1: 1}).len() == 1
    assert pnq([1, 2]).len() == 2
    assert pnq({1: 1, 2: 2}).len() == 2


class TestSkipTakeRangePage:
    def test_skip(self):
        assert pnq([]).skip(0).to_list() == []
        assert pnq([]).skip(1).to_list() == []
        assert pnq([1]).skip(0).to_list() == [1]
        assert pnq([1]).skip(1).to_list() == []
        assert pnq([1, 2]).skip(0).to_list() == [1, 2]
        assert pnq([1, 2]).skip(1).to_list() == [2]
        assert pnq([1, 2]).skip(2).to_list() == []

    def test_take(self):
        assert pnq([]).take(0).to_list() == []
        assert pnq([]).take(1).to_list() == []
        assert pnq([1]).take(0).to_list() == []
        assert pnq([1]).take(1).to_list() == [1]
        assert pnq([1, 2]).take(0).to_list() == []
        assert pnq([1, 2]).take(1).to_list() == [1]
        assert pnq([1, 2]).take(2).to_list() == [1, 2]

    def test_range(self):
        q = pnq([1, 2, 3, 4, 5, 6])

        assert q.range(0, -1).to_list() == []
        assert q.range(0, 0).to_list() == []

        assert q.range(-1, 0).to_list() == []
        assert q.range(0, 1).to_list() == [1]
        assert q.range(1, 2).to_list() == [2]
        assert q.range(2, 3).to_list() == [3]

        assert q.range(-2, 0).to_list() == []
        assert q.range(0, 2).to_list() == [1, 2]
        assert q.range(2, 4).to_list() == [3, 4]
        assert q.range(4, 6).to_list() == [5, 6]

        assert q.range(5, 6).to_list() == [6]
        assert q.range(5, 7).to_list() == [6]
        assert q.range(6, 7).to_list() == []

    def test_page(self):
        from pnq import Query

        with pytest.raises(ValueError):
            Query._page_calc(0, -1)

        assert Query._page_calc(-1, 0) == (0, 0)

        assert Query._page_calc(0, 0) == (0, 0)
        assert Query._page_calc(1, 0) == (0, 0)
        assert Query._page_calc(2, 0) == (0, 0)
        assert Query._page_calc(3, 0) == (0, 0)

        assert Query._page_calc(0, 1) == (-1, 0)
        assert Query._page_calc(1, 1) == (0, 1)
        assert Query._page_calc(2, 1) == (1, 2)
        assert Query._page_calc(3, 1) == (2, 3)

        assert Query._page_calc(0, 2) == (-2, 0)
        assert Query._page_calc(1, 2) == (0, 2)
        assert Query._page_calc(2, 2) == (2, 4)
        assert Query._page_calc(3, 2) == (4, 6)

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

        assert q.page(0, 0).to_list() == []
        assert q.page(1, 0).to_list() == []
        assert q.page(2, 0).to_list() == []
        assert q.page(3, 0).to_list() == []

        assert q.page(0, 1).to_list() == []
        assert q.page(1, 1).to_list() == [1]
        assert q.page(2, 1).to_list() == [2]
        assert q.page(3, 1).to_list() == [3]

        assert q.page(0, 2).to_list() == []
        assert q.page(1, 2).to_list() == [1, 2]
        assert q.page(2, 2).to_list() == [3, 4]
        assert q.page(3, 2).to_list() == [5, 6]


def test_indexing():
    def assert_query(db):
        assert db[1] == "a"
        assert db.get(1) == "a"
        assert db.get_or_default(1, 10) == "a"
        assert db.get_or_none(1) == "a"
        with pytest.raises(KeyError):
            assert db.get(-1)
        assert db.get_or_default(-1, 10) == 10
        assert db.get_or_none(-1) == None
        assert db.get_many(1, 2).to_list() == [(1, "a"), (2, "b")]
        assert db.get_many(4).to_list() == []
        assert db.keys().to_list() == [1, 2, 3]
        assert db.values().to_list() == ["a", "b", "c"]
        assert db.items().to_list() == [(1, "a"), (2, "b"), (3, "c")]

    db = pnq({1: "a", 2: "b", 3: "c"})
    assert_query(db)
    from_tuple = pnq([(1, "a"), (2, "b"), (3, "c")])
    assert_query(from_tuple.to_index())

    with pytest.raises(NotImplementedError):
        db.to_index()

    with pytest.raises(NotImplementedError):
        db.save()


def test_get():
    @block
    def case_no_elements():
        q = pnq([])  # type: ignore

        with pytest.raises(NoElementError):
            q.first()

        with pytest.raises(NoElementError):
            q.one()

        with pytest.raises(NoElementError):
            q.last()

        assert q.first_or_default() is None
        assert q.one_or_default() is None
        assert q.last_or_default() is None
        assert q.first_or_default(1) == 1
        assert q.one_or_default(2) == 2
        assert q.last_or_default(3) == 3

    @block
    def case_one_elements():
        q = pnq([5])

        assert q.first() == 5
        assert q.one() == 5
        assert q.last() == 5

        assert q.first_or_default() == 5
        assert q.one_or_default() == 5
        assert q.last_or_default() == 5
        assert q.first_or_default(1) == 5
        assert q.one_or_default(2) == 5
        assert q.last_or_default(3) == 5

    @block
    def case_two_elements():
        q = pnq([-10, 10])

        assert q.first() == -10
        with pytest.raises(NotOneError):
            q.one()
        assert q.last() == 10

        assert q.first_or_default() == -10
        assert q.one_or_default() is None
        assert q.last_or_default() == 10
        assert q.first_or_default(1) == -10
        assert q.one_or_default(2) == 2
        assert q.last_or_default(3) == 10


class Hoge:
    def __init__(self, id, name=""):
        self.id = id
        self.name = name


class TestSort:
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
        assert pnq([]).reverse().to_list() == []
        assert list(reversed(pnq([]))) == []
        assert pnq({}).reverse().to_list() == []
        assert list(reversed(pnq({}))) == []
        assert pnq([]).order_by_attrs("id").to_list() == []
        assert pnq([]).order_by_items("id").to_list() == []
        assert pnq([]).order(lambda x: x).to_list() == []

    def test_one_elements(self):
        obj = Hoge(id=10)

        assert pnq([1]).reverse().to_list() == [1]
        assert list(reversed(pnq([1]))) == [1]
        assert pnq({1: "a"}).reverse().to_list() == [(1, "a")]
        assert list(reversed(pnq({1: "a"}))) == [(1, "a")]
        assert pnq([obj]).order_by_attrs("id").to_list() == [obj]
        assert pnq([tuple([10])]).order_by_items(0).to_list() == [(10,)]
        assert pnq({1: "a"}).order_by_items(1).to_list() == [(1, "a")]
        assert pnq([obj]).order(lambda x: x.id).to_list() == [obj]

    def test_two_elements(self):
        obj1 = Hoge(id=10, name="b")
        obj2 = Hoge(id=20, name="a")

        assert pnq([2, 1]).reverse().to_list() == [1, 2]
        assert list(reversed(pnq([2, 1]))) == [1, 2]
        assert pnq({2: "a", 1: "a"}).reverse().to_list() == [(1, "a"), (2, "a")]
        assert list(reversed(pnq({2: "a", 1: "a"}))) == [(1, "a"), (2, "a")]
        assert pnq([obj2, obj1]).order_by_attrs("id").to_list() == [obj1, obj2]
        assert pnq([obj1, obj2]).order_by_attrs("id").to_list() == [obj1, obj2]
        assert pnq([(2, 0), (1, 100)]).order_by_items(0).to_list() == [(1, 100), (2, 0)]
        assert pnq([(2, 0), (1, 100)]).order_by_items(1).to_list() == [(2, 0), (1, 100)]
        assert pnq({2: "a", 1: "b"}).order_by_items(0).to_list() == [(1, "b"), (2, "a")]
        assert pnq({2: "a", 1: "b"}).order_by_items(1).to_list() == [(2, "a"), (1, "b")]
        assert pnq([obj2, obj1]).order(lambda x: x.id).to_list() == [obj1, obj2]
        assert pnq([obj2, obj1]).order(lambda x: x.name).to_list() == [obj2, obj1]

    def test_multi_value(self):
        obj1 = Hoge(id=10, name="b")
        obj2 = Hoge(id=20, name="a")
        obj3 = Hoge(id=30, name="a")

        assert pnq([obj3, obj2, obj1]).order(lambda x: (x.id, x.name)).to_list() == [
            obj1,
            obj2,
            obj3,
        ]
        assert pnq([obj3, obj2, obj1]).order(lambda x: (x.name, x.id)).to_list() == [
            obj2,
            obj3,
            obj1,
        ]

        assert pnq([obj3, obj2, obj1]).order_by_attrs("id", "name").to_list() == [
            obj1,
            obj2,
            obj3,
        ]
        assert pnq([obj3, obj2, obj1]).order_by_attrs("name", "id").to_list() == [
            obj2,
            obj3,
            obj1,
        ]

        dic1 = dict(id=10, name="b")
        dic2 = dict(id=20, name="a")
        dic3 = dict(id=30, name="a")

        assert pnq([dic3, dic2, dic1]).order_by_items("id", "name").to_list() == [
            dic1,
            dic2,
            dic3,
        ]
        assert pnq([dic3, dic2, dic1]).order_by_items("name", "id").to_list() == [
            dic2,
            dic3,
            dic1,
        ]


class TestSleep:
    def test_sync(self):
        pnq([1, 2, 3]).sleep(0).to_list() == [1, 2, 3]

    def test_async(self):
        import asyncio

        results = []

        async def func():
            async for elm in pnq([1, 2, 3]).asleep(0):
                results.append(elm)

        asyncio.run(func())
        assert results == [1, 2, 3]
