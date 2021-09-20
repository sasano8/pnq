import pnq


def mul(x):
    return x * 2


def test_async():
    assert pnq.query([1, 2, 3]).map(mul).test([2, 4, 6]) == (True, True)
