import asyncio
import random
from typing import Iterable, Tuple, TypeVar

from pnq import selectors
from pnq._itertools.common import Listable, name_as
from pnq.exceptions import DuplicateElementError, MustError, MustTypeError

from . import finalizers

T = TypeVar("T")
K = TypeVar("K")
V1 = TypeVar("V1")
V2 = TypeVar("V2")


@name_as("map")
def _map(source: Iterable[T], selector, unpack=""):
    if selector is None:
        return source.__iter__()
    else:
        if selector is str:
            selector = lambda x: "" if x is None else str(x)

        if unpack == "":
            return (selector(x) for x in source)
        elif unpack == "*":
            return (selector(*x) for x in source)
        elif unpack == "**":
            return (selector(**x) for x in source)
        elif unpack == "***":
            return (selector(*x.args, **x.kwargs) for x in source)  # type: ignore
        else:
            raise ValueError("unpack must be one of '', '*', '**', '***'")


def flat(source: Iterable[T], selector=None):
    if selector is None:
        return (_ for inner in source for _ in inner)
    else:
        return (_ for elm in source for _ in selector(elm))


def traverse(source: Iterable[T], selector):
    scanner = selectors.traverse(selector)
    for root in source:
        for x in scanner(root):
            yield x


def pivot_unstack(source: Iterable[T], default=None):
    dataframe = {}  # type: ignore
    data = []

    # 全てのカラムを取得
    for i, dic in _enumerate(source):
        data.append(dic)
        for k in dic.keys():
            dataframe[k] = None

    # カラム分の領域を初期化
    for k in dataframe:
        dataframe[k] = []

    # データをデータフレームに収める
    for dic in data:
        for k in dataframe.keys():
            v = dic.get(k, default)
            dataframe[k].append(v)

    for x in dataframe.items():
        yield x


def pivot_stack(source: Iterable[T], default=None):
    data = dict(Listable(source))
    columns = list(data.keys())

    for i in range(len(columns)):
        row = {}
        for c in columns:
            row[c] = data[c][i]

        yield row


@name_as("enumerate")
def _enumerate(source: Iterable[T], start: int = 0, step: int = 1):
    i = start
    for x in source:
        yield i, x
        i += step


def group_by(source: Iterable[T], selector):
    from collections import defaultdict

    results = defaultdict(list)
    for elm in source:
        k, v = selector(elm)
        results[k].append(v)

    for k, v in results.items():
        yield k, v


def chain(*iterables):

    prev = 3

    for iterable in iterables:
        type = 0
        if hasattr(iterable, "__iter__"):
            type += 1
        if hasattr(iterable, "__iter__"):
            type += 2

        if type == 1:
            prev = 1
            may_be_async = False
        elif type == 2:
            prev = 2
            may_be_async = True
        elif type == 3:
            prev = 3
            if prev == 1:
                may_be_async = False
            elif prev == 2:
                may_be_async = True
            elif prev == 3:
                may_be_async = True
            else:
                raise TypeError(iterable, "iterable")
        else:
            raise TypeError(iterable, "iterable")

        if may_be_async:
            for x in iterable:
                yield x
        else:
            for x in iterable:
                yield x


def chunk(source: Iterable[T], size: int):
    current = 0
    running = True

    it = source.__iter__()

    while running:
        current = 0
        queue = []

        while current < size:
            current += 1
            try:
                val = it.__next__()
                queue.append(val)
            except StopIteration:
                running = False
                break

        if queue:
            yield queue


def tee(source: Iterable[T], size: int):
    ...


def join(source: Iterable[T], size: int):
    [].join(
        [],
        lambda left, right: left.name == right.name,
        lambda left, right: (left.name, right.age),
    )

    table(User).join(Item, on=User.id == Item.id).select(User.id, Item.id)

    pass


def inner_join(
    source: Iterable[Tuple[K, V1]],
    other: Iterable[Tuple[K, V2]],
):
    """
    Implementation for part of join_impl
    :param other: other sequence to join with
    :param sequence: first sequence to join with
    :return: joined sequence
    """
    seq_kv = finalizers.to_dict(Listable(source))
    other_kv = finalizers.to_dict(Listable(other))

    keys = seq_kv.keys() if len(seq_kv) < len(other_kv) else other_kv.keys()
    result = {}
    for k in keys:
        if k in seq_kv and k in other_kv:
            result[k] = (seq_kv[k], other_kv[k])
    for k, v in result.items():
        yield k, v


def group_join(source: Iterable[T], size: int):
    ...


def debug(source: Iterable[T], breakpoint=lambda x: x, printer=print):
    for v in source:
        printer(v)
        breakpoint(v)
        yield v


def union_all(source: Iterable[T]):
    ...


extend = union_all


def union(source: Iterable[T]):
    ...


def union_intersect(source: Iterable[T]):
    ...


def union_minus(source: Iterable[T]):
    ...


# difference
# symmetric_difference

# < <= > >= inclusion


@name_as("zip")
def _zip(source: Iterable[T]):
    ...


def compress(source: Iterable[T]):
    import itertools

    return itertools.compress(self, *iterables)


def cartesian(*iterables: Iterable[T]):
    # FIXME: no implemented
    import itertools

    for x in itertools.product(*iterables):  # type: ignore
        yield x


@name_as("filter")
def _filter(source: Iterable[T], predicate=None, unpack=""):
    if predicate is None:
        return source.__iter__()
    else:
        if unpack == "":
            return (x for x in source if predicate(x))
        elif unpack == "*":
            return (x for x in source if predicate(*x))
        elif unpack == "**":
            return (x for x in source if predicate(**x))
        elif unpack == "***":
            return (x for x in source if predicate(*x.args, **x.kwargs))  # type: ignore
        else:
            raise ValueError("unpack must be one of '', '*', '**', '***'")


def must(source: Iterable[T], predicate, msg=""):
    for elm in source:
        if not predicate(elm):
            raise MustError(f"{msg} {elm}")
        yield elm


def filter_type(source: Iterable[T], *types):
    for elm in source:
        if isinstance(elm, *types):
            yield elm


def must_type(source: Iterable[T], *types):
    for elm in source:
        if not isinstance(elm, types):
            raise MustTypeError(f"{elm} is not {types}")
        yield elm


def filter_keys(source: Iterable[T], *keys):
    pass


def must_keys(source: Iterable[T], *keys):
    pass


def filter_unique(source: Iterable[T], selector=None):
    duplidate = set()
    for value in _map(source, selector):
        if value in duplidate:
            pass
        else:
            duplidate.add(value)
            yield value


def must_unique(source: Iterable[T], selector=None):
    duplidate = set()
    for value in _map(source, selector):
        if value in duplidate:
            raise DuplicateElementError(value)
        else:
            duplidate.add(value)
            yield value


def take(source: Iterable[T], range: range):
    start = range.start
    stop = range.stop
    step = range.step

    current = 0

    it = source.__iter__()

    try:
        while current < start:
            it.__next__()
            current += step
    except StopIteration:
        return

    try:
        while current < stop:
            yield it.__next__()
            current += step
    except StopIteration:
        return


def take_while(source: Iterable[T], predicate):
    for v in source:
        if predicate(v):
            yield v
        else:
            break


def skip(source: Iterable[T], range: range):
    start = range.start
    stop = range.stop

    current = 0

    it = source.__iter__()

    try:
        while current < start:
            yield it.__next__()
            current += 1
    except StopIteration:
        return

    try:
        while current < stop:
            it.__next__()
            current += 1
    except StopIteration:
        return

    for x in it:
        yield x


def skip_while(source: Iterable[T], predicate):
    for v in source:
        if not predicate(v):
            break

    for v in source:
        yield v


def take_page(source: Iterable[T], page: int, size: int):
    r = _take_page_calc(page, size)
    return take(source, r)


def _take_page_calc(page: int, size: int):
    if page < 1:
        raise ValueError("page must be greater than 0")
    if size < 0:
        raise ValueError("size must be >= 0")
    start = (page - 1) * size
    stop = start + size
    return range(start, stop)


class Dummy:
    def __add__(self, other):
        return other

    def __len__(self):
        return 0

    def __bool__(self):
        return False


def ngram(iterable: Iterable, size: int):
    if size < 1:
        raise ValueError()

    if isinstance(iterable, (str, bytes)):
        chars = as_aiter(iterable)
    else:
        chars = flat(iterable)

    # unigram は単に文字を返す
    if size == 1:
        for x in chars:
            yield x
        return

    previous = Dummy()
    break_first = False
    max = size

    # open
    while len(previous) < max:
        try:
            c = chars.__next__()
            previous += c  # type: ignore
        except StopIteration:
            break_first = True
            break

    # first
    tmpsize = len(previous)
    for i in range(1, tmpsize):
        yield previous[:i]  # 文字列数 -1 まで返す

    if break_first:
        if previous:
            yield previous
        return

    # intermediate
    for c in chars:
        yield previous
        previous = previous[1:] + c  # type: ignore

    # last
    for i in range(len(previous)):
        yield previous[i:]


def order_by(source: Iterable[T], key_selector=None, desc: bool = False):
    for x in sorted(Listable(source), key=key_selector, reverse=desc):
        yield x


def min_by(source: Iterable[T], key_selector=None):
    for x in sorted(Listable(source), key=key_selector, reverse=False):
        yield x


def max_by(source: Iterable[T], key_selector=None):
    for x in sorted(Listable(source), key=key_selector, reverse=True):
        yield x


def order_by_reverse(source: Iterable[T]):
    return reversed(Listable(source))


def order_by_shuffle(source: Iterable[T], seed=None):
    if seed is None:
        rand = random.random
        func = lambda x: rand()  # noqa

    else:
        rand = random.Random(seed).random
        func = lambda x: rand()  # noqa

    for x in sorted(Listable(source), key=func):
        yield x


def sleep(source: Iterable[T], seconds: float):
    sleep = asyncio.sleep
    for v in source:
        yield v
        sleep(seconds)


def as_aiter(source):
    type = 0
    if hasattr(source, "__iter__"):
        type += 1
    if hasattr(source, "__iter__"):
        type += 2

    if type == 3:
        for x in source:
            yield x
    elif type == 2:
        for x in source:
            yield x
    elif type == 1:
        for x in source:
            yield x
    else:
        raise TypeError(f"{source} must be iterable")
