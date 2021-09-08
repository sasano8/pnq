from pnq.interfaces import DictEx, ListEx, PairQuery


def playground():
    a = DictEx({1: "a"}).to_index()
    b: PairQuery[str, int] = cast(PairQuery[str, int], PairQuery())  # type: ignore

    reveal_type(b)
    c = b.to_index()

    reveal_type(c)

    q1 = ListEx([1]).pairs(lambda x: (1, "aa"))
    reveal_type(q1)

    q2 = q1.filter(lambda x: x[0] == 1)
    reveal_type(q2)

    reveal_type(q2.to_index)

    q3 = q2.to_index()
    reveal_type(q3)

    q5 = q2.to_list()
    reveal_type(q5)

    q4 = q2.to_list()
    reveal_type(q4)


def test_debug():
    assert ListEx((1, 2, 3)).map(lambda x: x * 2).debug() == "[1, 2, 3].map((x) => ...)"
    ListEx((1, 2, 3)).range(1, 4).range(1, 4).map(lambda x: x * 2).map(
        str
    ).debug() == "[1, 2, 3].range(1, 4).range(1, 4).map((x) => ...).map(str)"


def test_init():
    assert ListEx() == []
    assert DictEx() == {}
    assert ListEx((1, 2)) == [1, 2]
    assert DictEx([(1, "a"), (2, "b")]) == {1: "a", 2: "b"}


def test_map():
    assert ListEx([1]).map(lambda x: x * 2).to_list() == [2]
    assert DictEx({1: "a"}).map(lambda x: (x[0] + 1, x[1] + "b")).to_list() == [
        (2, "ab")
    ]

    aaa = (
        DictEx({1: "a"})
        .map(lambda x: (x[0] + 1, x[1] + "b"))
        .pairs()
        .map(lambda x: (x[0], x[1]))
        .to_index()
        .get_many(1, 2, 3)
        .reverse()
    )
