def test_mypy():
    from typing import Iterable, Tuple, TypeVar

    from pnq.types import query  # type: ignore

    T = TypeVar("T")

    q = query([1, 2, 3]).filter(lambda x: x > 1)
    arr = q.to_list()

    def func(a: Iterable[T]) -> T:
        ...

    b = [1, 2, 3]

    result = func(q)
    print(result)

    query({1: "a"}).select(1).take(3).to_list()

    a = query({1: "a"}).to_dict()

    from pnq import query

    result = query([]).cast(Tuple[int, str]).to_dict()
    result = query([]).cast(lambda x: (1, "")).to_dict()
    result = query([]).map(lambda x: (1, 2)).to_dict()
