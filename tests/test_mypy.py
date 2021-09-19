def test_mypy():
    from typing import Iterable, Tuple, TypeVar

    import pnq as pq
    from pnq.queries import query  # type: ignore

    T = TypeVar("T")

    q = query([1, 2, 3]).filter(lambda x: x > 1)
    arr = q.to(list)
    print(arr)

    def func(a: Iterable[T]) -> T:
        ...

    b = [1, 2, 3]

    result = func(q)
    print(result)

    query({1: "a"}).select(1).take(3).to(list)

    a = query({1: "a"}).to(dict)

    from pnq import query

    result = query([]).cast(Tuple[int, str]).to(dict)
    result = query([]).cast(lambda x: (1, "")).to(dict)
    result = query([]).map(lambda x: (1, 2)).to(dict)
