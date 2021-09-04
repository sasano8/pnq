import pytest

from pnq import pnq
from pnq.exceptions import NoElementError, NotOneError


def block(func):
    func()
    return func


def test_pnq():
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
    assert pnq([1]).map(lambda x: x * 2).to_list() == [2]
    assert pnq([1]).filter(lambda x: x == 1).to_list() == [1]
    assert pnq([1]).filter(lambda x: x == 0).to_list() == []
    assert pnq([1]).filter(lambda x: x == 0).to_list() == []
    assert pnq([1]).enumerate().to_list() == [(0, 1)]
    # assert pnq([{"key": 3, "name": "test"}]).lookup(lambda x: x["key"]).to_list() == [
    #     (3, {"key": 3, "name": "test"})
    # ]
    # obj = pnq([{"key": 3, "name": "test"}]).lookup(lambda x: x["key"]).to_dict()
    # assert obj[3] == {"key": 3, "name": "test"}


def test_aggregator():
    assert pnq([]).len() == 0
    assert pnq({}).len() == 0
    assert pnq([1]).len() == 1
    assert pnq({1: 1}).len() == 1
    assert pnq([1, 2]).len() == 2
    assert pnq({1: 1, 2: 2}).len() == 2


def test_slice():
    assert pnq([]).slice(0, 100).to_list() == []
    assert pnq([1]).slice(0, 0).to_list() == []
    assert pnq([1]).slice(0, 1).to_list() == [1]
    assert pnq([1]).slice(0).to_list() == [1]
    assert pnq([1]).slice(-1, 1).to_list() == [1]
    assert pnq([1, 2]).slice(0, 1).to_list() == [1]
    assert pnq([1, 2, 3]).slice(1, 1).to_list() == []
    assert pnq([1, 2, 3]).slice(1, 2).to_list() == [2]
    assert pnq([1, 2, 3]).slice(1, 3).to_list() == [2, 3]
    assert pnq([1, 2, 3]).slice(1).to_list() == [2, 3]


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


def test_unpack():
    with pytest.raises(TypeError, match="missing 1 required positional argument: 'v'"):
        pnq([(1, 2)]).map(lambda k, v: (k, v)).to_dict()
    assert pnq([(1, 2)]).map_unpack(lambda k, v: (k, v)).to_dict() == {1: 2}
    assert pnq([{"name": "test", "age": 20}]).map_unpack_kw(
        lambda name, age: (name, age)
    ).to_list() == [("test", 20)]


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


def test_sort():
    @block
    def case_no_elements():
        assert pnq([]).reverse().to_list() == []
        assert list(reversed(pnq([]))) == []
        assert pnq({}).reverse().to_list() == []
        assert list(reversed(pnq({}))) == []
        assert pnq([]).order_by_attrs("id").to_list() == []
        assert pnq([]).order_by_index("id").to_list() == []
        assert pnq([]).order(lambda x: x).to_list() == []

    class Hoge:
        def __init__(self, id, name=""):
            self.id = id
            self.name = name

    @block
    def case_one_elements():
        obj = Hoge(id=10)

        assert pnq([1]).reverse().to_list() == [1]
        assert list(reversed(pnq([1]))) == [1]
        assert pnq({1: "a"}).reverse().to_list() == [(1, "a")]
        assert list(reversed(pnq({1: "a"}))) == [(1, "a")]
        assert pnq([obj]).order_by_attrs("id").to_list() == [obj]
        assert pnq([tuple([10])]).order_by_index(0).to_list() == [(10,)]
        assert pnq({1: "a"}).order_by_index(1).to_list() == [(1, "a")]
        assert pnq([obj]).order(lambda x: x.id).to_list() == [obj]

    @block
    def case_two_elements():
        obj1 = Hoge(id=10, name="b")
        obj2 = Hoge(id=20, name="a")

        assert pnq([2, 1]).reverse().to_list() == [1, 2]
        assert list(reversed(pnq([2, 1]))) == [1, 2]
        assert pnq({2: "a", 1: "a"}).reverse().to_list() == [(1, "a"), (2, "a")]
        assert list(reversed(pnq({2: "a", 1: "a"}))) == [(1, "a"), (2, "a")]
        assert pnq([obj2, obj1]).order_by_attrs("id").to_list() == [obj1, obj2]
        assert pnq([obj1, obj2]).order_by_attrs("id").to_list() == [obj1, obj2]
        assert pnq([(2, 0), (1, 100)]).order_by_index(0).to_list() == [(1, 100), (2, 0)]
        assert pnq([(2, 0), (1, 100)]).order_by_index(1).to_list() == [(2, 0), (1, 100)]
        assert pnq({2: "a", 1: "b"}).order_by_index(0).to_list() == [(1, "b"), (2, "a")]
        assert pnq({2: "a", 1: "b"}).order_by_index(1).to_list() == [(2, "a"), (1, "b")]
        assert pnq([obj2, obj1]).order(lambda x: x.id).to_list() == [obj1, obj2]
        assert pnq([obj2, obj1]).order(lambda x: x.name).to_list() == [obj2, obj1]

    @block
    def case_multi_value():
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

        assert pnq([dic3, dic2, dic1]).order_by_index("id", "name").to_list() == [
            dic1,
            dic2,
            dic3,
        ]
        assert pnq([dic3, dic2, dic1]).order_by_index("name", "id").to_list() == [
            dic2,
            dic3,
            dic1,
        ]
