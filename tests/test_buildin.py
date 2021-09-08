from pnq.buildins import pfilter, piter, plen, pmap
from pnq.protocol import KeyValueItems


def test_len():
    assert plen([]) == 0
    assert plen([1]) == 1


def test_piter():
    assert list(piter([])) == []
    assert list(piter([1])) == [1]
    assert list(piter({})) == []
    assert list(piter({1: "a"})) == [(1, "a")]


def test_pfilter():
    assert list(pfilter([], lambda x: True)) == []
    assert list(pfilter([1], lambda x: True)) == [1]
    assert list(pfilter({}, lambda x: True)) == []
    assert list(pfilter({1: "a"}, lambda x: True)) == [(1, "a")]

    assert list(pfilter([1], lambda x: False)) == []
    assert list(pfilter({1: "a"}, lambda x: False)) == []


def test_pmap():
    assert list(pmap([])) == []
    assert list(pmap([], lambda x: True)) == []
    assert list(pmap([1])) == [1]
    assert list(pmap([1], lambda x: True)) == [True]

    assert list(pmap({})) == []
    assert list(pmap({}, lambda x: True)) == []
    assert list(pmap({1: "a"})) == [(1, "a")]
    assert list(pmap({1: "a"}, lambda x: True)) == [True]


def test_protocol():
    assert isinstance(dict, KeyValueItems)
