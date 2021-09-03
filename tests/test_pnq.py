import pytest

from pnq import pnq


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
    assert pnq([1]).enumrate().to_list() == [(0, 1)]
    assert pnq([{"key": 3, "name": "test"}]).lookup(lambda x: x["key"]).to_list() == [
        (3, {"key": 3, "name": "test"})
    ]
    obj = pnq([{"key": 3, "name": "test"}]).lookup(lambda x: x["key"]).to_dict()
    assert obj[3] == {"key": 3, "name": "test"}


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
