import statistics
from asyncio import sleep as asleep
from collections import defaultdict
from functools import wraps
from itertools import zip_longest
from time import sleep
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Generic,
    Iterable,
    Iterator,
    List,
    Mapping,
    Tuple,
    Type,
    TypeVar,
    Union,
    overload,
)

from . import reflectors
from .exceptions import NoElementError, NotOneError
from .getter import attrgetter, itemgetter

T = TypeVar("T")
K = TypeVar("K")
V = TypeVar("V")
R = TypeVar("R")
R2 = TypeVar("R2")
T_OTHER = TypeVar("T_OTHER")
F = TypeVar("F", bound=Callable)
undefined = object()


class Lazy(Generic[T]):
    def __init__(self, func: Callable[..., Iterator[T]], *args, **kwargs):
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def __call__(self) -> Iterator[T]:
        return self.func(*self.args, **self.kwargs)

    def __iter__(self) -> Iterator[T]:
        yield from self()

    @classmethod
    def decolate(cls, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return cls(func, *args, **kwargs)

        return wrapper


class AsyncLazy(Generic[T]):
    def __init__(self, func: Callable[..., AsyncIterator[T]], *args, **kwargs):
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def __call__(self) -> AsyncIterator[T]:
        return self.func(*self.args, **self.kwargs)

    async def __aiter__(self) -> AsyncIterator[T]:
        async for elm in self():
            yield elm

    @classmethod
    def decolate(cls, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return cls(func, *args, **kwargs)

        return wrapper


lazy = Lazy.decolate
alazy = AsyncLazy.decolate


class Transform(Iterable[T]):
    pass


dummy = lambda x: x


class Query(Iterable[T]):
    def __init__(self, source: Iterable[T]):
        if isinstance(source, Mapping):
            raise TypeError()

        self.source = source

    def __iter__(self) -> Iterator[T]:
        return self.source.__iter__()

    def len(self):
        return len(list(self))

    @staticmethod
    def _first(source):
        it = iter(source)
        try:
            obj = next(it)
        except StopIteration:
            raise NoElementError()

        return it, obj

    @staticmethod
    def _last(source: Iterable):
        undefined = object()
        last = undefined
        for elm in source:
            last = elm

        if last is undefined:
            raise NoElementError()

        return last

    def first(self):
        it, obj = self._first(self)
        return obj

    def first_or_default(self, default: R = None) -> Union[T, R, None]:
        try:
            return self.first()
        except NoElementError:
            return default

    def one(self) -> T:
        it, obj = self._first(self)
        try:
            next(it)
        except StopIteration:
            return obj

        raise NotOneError()

    def one_or_default(self, default: R = None) -> Union[T, R, None]:
        try:
            return self.one()
        except NotOneError:
            return default
        except NoElementError:
            return default

    def last(self):
        return self._last(self)

    def last_or_default(self, default: R = None) -> Union[T, R, None]:
        try:
            return self.last()
        except NoElementError:
            return default

    @staticmethod
    @lazy
    def _order(source, selector, desc: bool = False):
        yield from sorted(source, key=selector, reverse=desc)

    def order(self: Iterable[T], selector, desc: bool = False) -> "Query[T]":
        return Query(Query._order(self, selector, desc))

    def order_by_items(
        self: Iterable[T], *items: Any, desc: bool = False
    ) -> "Query[T]":
        selector = itemgetter(*items)
        return Query.order(self, selector, desc=desc)

    def order_by_attrs(
        self: Iterable[T], *attrs: str, desc: bool = False
    ) -> "Query[T]":
        selector = attrgetter(*attrs)
        return Query.order(self, selector, desc=desc)

    def __reversed__(self: Iterable[T]):
        arr = list(self)
        yield from reversed(arr)

    @staticmethod
    @lazy
    def _reverse(source: Iterable[T]) -> Iterator[T]:
        yield from Query.__reversed__(source)

    def reverse(self: Iterable[T]) -> "Query[T]":
        return Query(Query._reverse(self))

    @staticmethod
    @lazy
    def _enumerate(source: Iterable[T], start: int = 0) -> Iterator[Tuple[int, T]]:
        yield from enumerate(source, start)

    def enumerate(self: Iterable[T], start: int = 0) -> "Query[Tuple[int, T]]":
        return Query(Query._enumerate(self, start))

    @staticmethod
    @lazy
    def _filter(source, func):
        return filter(func, source)

    def filter(self, func) -> "Query[T]":
        return Query(Query._filter(self, func))

    def filter_type(self: Iterable[T], *types: Type[R]) -> "Query[R]":
        def normalize(type):
            if type is None:
                return None.__class__
            else:
                return type

        types = tuple((normalize(x) for x in types))
        return Query(Query._filter(self, lambda x: isinstance(x, *types)))

    @staticmethod
    @lazy
    def _map(source: Iterable[T], selector: Callable[[T], R]) -> Iterator[R]:
        return map(selector, source)

    @staticmethod
    @lazy
    def _unpack(source: Iterable[T], selector: Callable[..., R]) -> Iterator[R]:
        for elm in source:
            yield selector(*elm)  # type: ignore

    @staticmethod
    @lazy
    def _unpack_kw(source: Iterable[T], selector: Callable[..., R]) -> Iterator[R]:
        for elm in source:
            yield selector(**elm)  # type: ignore

    def map(self, selector: Callable[[T], R]) -> "Query[R]":
        if selector is str:
            selector = reflectors.to_str
        return Query(Query._map(self, selector))

    def unpack(self, selector: Callable[..., R]) -> "Query[R]":
        return Query(Query._unpack(self, selector))

    def unpack_kw(self, selector: Callable[..., R]) -> "Query[R]":
        return Query(Query._unpack_kw(self, selector))

    def select(self, item) -> "Query[Any]":
        return Query(Query._map(self, lambda x: x[item]))

    def select_item(self, item) -> "Query[Any]":
        return Query.select(self, item)

    def select_attr(self, attr: str) -> "Query[Any]":
        return Query(Query._map(self, lambda x: getattr(x, attr)))

    def select_attrs(self, *attrs: Any) -> "Query[Tuple]":
        if len(attrs) == 0:
            selector = lambda x: tuple()  # type: ignore
        elif len(attrs) == 1:
            # itemgetter/getattrは引数が１の時、タプルでなくそのセレクトされた値を直接返のでタプルで返すようにする
            name = attrs[0]
            selector = lambda x: (getattr(x, name),)
        else:
            selector = attrgetter(*attrs)

        return Query(Query._map(self, selector))

    def select_items(self, *items: str) -> "Query[Tuple]":
        if len(items) == 0:
            selector = lambda x: tuple()  # type: ignore
        elif len(items) == 1:
            # itemgetter/getattrは引数が１の時、タプルでなくそのセレクトされた値を直接返のでタプルで返すようにする
            name = items[0]
            selector = lambda x: (x[name],)
        else:
            selector = itemgetter(*items)
        return Query(Query._map(self, selector))

    def cast(self: "Query[T]", type: Type[R]) -> "Query[R]":
        return self  # type: ignore

    def reduce(
        self: Iterable[T],
        accumulator: Callable[[T, T], T],
        seed: T = undefined
        # self: Iterable[T], accumulator: Callable[[R, R], R], seed: R = undefined
    ) -> R:
        from functools import reduce

        if seed is undefined:
            return reduce(accumulator, self)
        else:
            return reduce(accumulator, self, seed)

    @staticmethod
    @lazy
    def _must_unique(source: Iterable[T], selector: Callable[[T], R]):
        seen = set()
        duplicated = []
        for elm in source:
            if selector(elm) in seen:
                duplicated.append(elm)
            else:
                seen.add(selector(elm))

        if duplicated:
            raise ValueError(f"Duplicated elements: {duplicated}")

        for elm in source:
            yield elm

    def must_unique(
        self: Iterable[T], selector: Callable[[T], R] = lambda x: x
    ) -> "Query[T]":
        return Query(set(self))

    def zip(
        self: Iterable[T],
        *others: T_OTHER,
        short: bool = False,
        fillvalue: Any = undefined,
    ) -> "Query[Tuple]":
        raise NotImplementedError()

    @staticmethod
    @lazy
    def _skip(self, count: int):
        it = iter(self)
        current = 0

        try:
            while current < count:
                next(it)
                current += 1
        except StopIteration:
            return

        for elm in it:
            yield elm

    def skip(self, count: int):
        return Query(Query._skip(self, count))

    @staticmethod
    @lazy
    def _take(self, count: int):
        it = iter(self)
        current = 0

        try:
            while current < count:
                yield next(it)
                current += 1
        except StopIteration:
            return

    def take(self, count: int):
        return Query(Query._take(self, count))

    @staticmethod
    @lazy
    def _range(self, start: int = 0, stop: int = None):
        it = iter(self)
        if start < 0:
            start = 0

        if stop is None:
            stop = float("inf")  # type: ignore
        elif stop < 0:
            stop = 0
        else:
            pass

        current = 0

        try:
            while current < start:
                next(it)
                current += 1
        except StopIteration:
            return

        try:
            while current < stop:  # type: ignore
                yield next(it)
                current += 1
        except StopIteration:
            return

    def range(self, start: int = 0, stop: int = None) -> "Query[T]":
        return Query(Query._range(self, start, stop))

    @staticmethod
    def _page_calc(page: int, size: int):
        if size < 0:
            raise ValueError("size must be >= 0")
        start = (page - 1) * size
        stop = start + size
        return start, stop

    def page(self, page: int = 1, size: int = 0) -> "Query[T]":
        start, stop = Query._page_calc(page, size)
        return Query.range(self, start, stop)

    start = 0 * 1
    stop = 0 + 50 - 1

    def save(self):
        return Query(list(self))

    def save_index(self):
        raise NotImplementedError()

    def to_index(self: Iterable[Tuple[K, V]]) -> "QuerableDict[K, V]":
        source = {k: v for k, v in self}
        return QuerableDict(source)

    def to_list(self):
        return list(iter(self))

    def to_dict(self: Iterable[Tuple[K, V]]):
        return dict(iter(self))

    def to_lookup(
        self, selector: Callable[[T], R] = None
    ) -> "QuerableDict[R, Iterable[T]]":
        # 指定したキーを集約する
        raise NotImplementedError()

    @staticmethod
    @lazy
    def _group_by(
        source: Iterable[T], selector: Callable[[T], Tuple[K, V]]
    ) -> "Iterable[Tuple[K, List[V]]]":
        results: Dict[K, List[V]] = defaultdict(list)
        for elm in source:
            k, v = selector(elm)
            results[k].append(v)

        for k, v in results.items():  # type: ignore
            yield k, v  # type: ignore

    def group_by(
        self: Iterable[T], selector: Callable[[T], Tuple[K, V]] = lambda kv: kv  # type: ignore
    ):
        # to_lookupとあまり変わらないが
        # to_lookupは即時実行groupingが遅延評価
        return QuerableDictAdapter(Query._group_by(self, selector))

    def aggregate(self):
        raise NotImplementedError()

    @staticmethod
    @lazy
    def _sleep(source: Iterable[T], seconds: float):
        for elm in source:
            yield elm
            sleep(seconds)

    def sleep(self: Iterable[T], seconds: float = 0) -> "Query[T]":
        return Query(Query._sleep(self, seconds))

    @staticmethod
    @alazy
    async def _asleep(source: Iterable[T], seconds: float):
        for elm in source:
            yield elm
            await asleep(seconds)

    def asleep(self: Iterable[T], seconds: float = 0) -> AsyncLazy[T]:
        return Query._asleep(self, seconds)


class QueryTuple(Query[Tuple[K, V]], Iterable[Tuple[K, V]], Generic[K, V]):
    pass


class QuerableDictAdapter(QueryTuple[K, V]):
    def __init__(self, source: Iterable[Tuple[K, V]]):
        self.source = source

    def __getitem__(self, key):
        source = {k: v for k, v in self}
        return source[key]

    # 共通
    def __iter__(self) -> Iterator[Tuple[K, V]]:
        yield from ((k, v) for k, v in self.source)

    @staticmethod
    @lazy
    def _keys(self):
        yield from (k for k, v in self)

    @staticmethod
    @lazy
    def _values(self):
        yield from (v for k, v in self)

    @staticmethod
    @lazy
    def _items(self):
        raise NotImplementedError()

    def keys(self) -> Query[K]:
        return Query(QuerableDictAdapter._keys(self))

    def values(self) -> Query[V]:
        return Query(QuerableDictAdapter._values(self))

    def items(self) -> "QuerableDictAdapter[K, V]":
        return self


class DictWrapper:
    def __init__(self, source):
        self.source = source

    def __iter__(self) -> Iterator[Tuple[K, V]]:
        return self.source.items().__iter__()

    def __len__(self):
        return len(self.source)

    def __reversed__(self):
        source: dict = self.source
        for key in source.__reversed__():
            yield key, source[key]

    def keys(self):
        return self.source.keys()

    def values(self):
        return self.source.values()

    def items(self):
        return self.source.items()

    def __getitem__(self, key):
        return self.source[key]


class QuerableDict(QueryTuple[K, V]):
    def __init__(self, source):
        if isinstance(source, QuerableDict):
            self.source = source
        elif isinstance(source, QuerableDictAdapter):
            self.source = source
        elif isinstance(source, DictWrapper):
            self.source = source
        elif isinstance(source, Mapping):
            self.source = DictWrapper(source)
        else:
            # self.source = {k: v for k, v in source}
            raise TypeError("source must be a mapping")

    def len(self):
        if hasattr(self.source, "__len__"):
            return len(self.source)
        else:
            return len(dict(self.source))

    def __reversed__(self):
        return self.source.__reversed__()

    def save(self):
        raise NotImplementedError()

    def to_index(self):
        if isinstance(self, QuerableDict):
            return self
        elif isinstance(self, Mapping):
            return QuerableDict(self)
        else:
            raise TypeError()

    def __getitem__(self, key):
        return self.source[key]  # type: ignore

    # 共通
    def __iter__(self) -> Iterator[Tuple[K, V]]:
        yield from ((k, v) for k, v in self.source)

    @staticmethod
    @lazy
    def _keys(self):
        yield from (k for k, v in self)

    @staticmethod
    @lazy
    def _values(self):
        yield from (v for k, v in self)

    @staticmethod
    @lazy
    def _items(self):
        raise NotImplementedError()

    def keys(self) -> Query[K]:
        return Query(QuerableDictAdapter._keys(self))

    def values(self) -> Query[V]:
        return Query(QuerableDictAdapter._values(self))

    def items(self) -> "QuerableDict[K, V]":
        return self

    def get(self, key) -> V:
        return self.__getitem__(key)

    def get_or_none(self, key) -> Union[V, None]:
        return self.get_or_default(key, None)

    def get_or_default(self, key, default: R = None) -> Union[V, R, None]:
        try:
            return self.get(key)
        except KeyError:
            return default

    def get_many(self, *keys, duplicate: bool = True) -> Query[Tuple[K, V]]:
        keys = set(keys)  # type: ignore
        return QuerableDictAdapter(QuerableDict._get_many(self, keys))

    def get_many_or_raise(self, *keys, duplicate: bool = True) -> Query[Tuple[K, V]]:
        raise NotImplementedError()

    @staticmethod
    @lazy
    def _get_many(source: "QuerableDict", keys: Iterable) -> Iterator:
        source = source.to_index()
        for id in keys:
            obj = source.get_or_default(id, undefined)
            if not obj is undefined:
                yield id, obj


@overload
def pnq(source: Mapping[K, V]) -> QuerableDict[K, V]:
    ...


@overload
def pnq(source: Iterable[Tuple[K, V]]) -> QueryTuple[K, V]:
    ...


@overload
def pnq(source: Iterable[T]) -> Query[T]:
    ...


def pnq(  # type: ignore
    iterable_or_mapping: Union[Mapping[K, V], Iterable[Tuple[K, V]], Iterable[T]]
) -> Union[QuerableDict[K, V], Query[T]]:
    """
    シーケンス操作のAPIを備えたクエリオブジェクトを返します。
    `Mapping[K, V]`を渡した場合、クエリオブジェクトの`\_\_iter\_\_`は`Tuple[K, V]`を
    返します。

    **Parameters:**

    * **iterable_or_mapping** - `Query` オブジェクトが操作するソース

    **Returns:** `Query`

    Usage:
    ```
    >>> from pnq import pnq
    >>> iterable = pnq([1])
    >>> iterable_key_value = pnq({1, "a"})
    >>> [x for x in iterable]
    [1]
    >>> [x for x in iterable_key_value]
    [(1, "a")]
    ```
    """
    if isinstance(iterable_or_mapping, Mapping):
        return QuerableDict(iterable_or_mapping)
    else:
        return Query(iterable_or_mapping)  # type: ignore


class Pnq:
    """シーケンス走査APIを備えたクエリオブジェクト

    Usage:
    ```
    >>> from pnq import pnq
    >>> iterable = pnq([1])
    >>> iterable_key_value = pnq({1, "a"})
    >>> [x for x in iterable]
    [1]
    >>> [x for x in iterable_key_value]
    [(1, "a")]
    ```
    """

    def query(iterable_or_mapping) -> "Pnq":
        """シーケンス走査APIを備えたクエリオブジェクトを返します。
        Mapping[K, V]を渡した場合、クエリオブジェクトの__iter__はTuple[K, V]を 返します。"""
        pass

    def filter(self, predicate):
        """述語に基づいて値のシーケンスをフィルター処理します。

        **Parameters:**

        * **self** - `Query` オブジェクト自身またはIterable[T]
        * **predicate: Callable[[T], bool]** - 各要素に対する検証関数

        **Returns:** `Query` | `QueryDict[K, V]`

        Usage:
        ```
        >>> import pnq
        >>> pnq.query([1, 2]).filter(lambda x: x == 1)
        [1]
        ```
        """
        pass

    def map(self, selector):
        """シーケンスの各要素を新しいフォームに射影します。
        また、利便性のたにstr関数を渡した場合のみ、Pythonの標準と異なる挙動をします（Usageを参照）。

        **Parameters:**

        * **self** - `Query` オブジェクト自身またはIterable[T]
        * **selector: Callable[[T], R]** - 各要素に対する変換関数


        **Returns:** `Query` | `QueryDict[K, V]`

        Usage:
        ```
        >>> import pnq
        >>> pnq.query([1]).map(lambda x: x * 2)
        [2]
        >>> pnq.query([None]).map(str)
        [""]
        ```
        """
        pass

    def unpack(self, selector):
        """シーケンスの各要素をアンパックし、新しいフォームに射影します。

        **Parameters:**

        * **self** - `Query` オブジェクト自身またはIterable[T]
        * **selector: Callable[[...], R]** - 各要素に対する変換関数

        **Returns:** `Query` | `QueryDict[K, V]`

        Usage:
        ```
        >>> import pnq
        >>> pnq.query([(1, 2)]).unpack(lambda a1, *_: a1)
        [1]
        ```
        """
        pass

    def unpack_kw(self, selector):
        """シーケンスの各要素をキーワードアンパックし、新しいフォームに射影します。

        **Parameters:**

        * **self** - `Query` オブジェクト自身またはIterable[T]
        * **selector: Callable[[...], R]** - 各要素に対する変換関数

        **Returns:** `Query` | `QueryDict[K, V]`

        Usage:
        ```
        >>> import pnq
        >>> pnq.query([dict(id=1, name="a")]).unpack_kw(lambda name, **_: name)
        ["a"]
        >>> pnq.query([dict(id=1, name="a")]).unpack_kw(lambda name, **_: locals())
        [{"a": "a", "_": 1}]
        ```
        """
        pass

    def select(self, item):
        """シーケンスの各要素からひとつのアイテムを選択し新しいフォームに射影します。

        **Parameters:**

        * **self** - `Query` オブジェクト自身またはIterable[T]
        * **item: str** - 選択するアイテム

        **Returns:** `Query` | `QueryDict[K, V]`

        Usage:
        ```
        >>> import pnq
        >>> pnq.query([(1, 2)]).select(0)
        [1]
        >>> pnq.query([dict(id=1, name="a", age=5)]).select("name")
        ["a"]
        ```
        """
        pass

    def select_attr(self, item):
        """シーケンスの各要素からひとつの属性を選択し新しいフォームに射影します。

        **Parameters:**

        * **self** - `Query` オブジェクト自身またはIterable[T]
        * **item: str** - 選択する属性

        **Returns:** `Query` | `QueryDict[K, V]`

        Usage:
        ```
        >>> import pnq
        >>> pnq.query([(str)]).select_attr("__name__")
        ["name"]
        ```
        """
        pass

    def select_items(self, *items):
        """シーケンスの各要素から複数のアイテムを選択し新しいフォームに射影します。

        **Parameters:**

        * **self** - `Query` オブジェクト自身またはIterable[T]
        * **\*items: str** - 選択するアイテム

        **Returns:** `Query` | `QueryDict[K, V]`

        Usage:
        ```
        >>> import pnq
        >>> pnq.query([dict(id=1, name="a", age=5)]).select_items("name", "age")
        [("a", 5)]
        ```
        """
        pass

    def select_attrs(self, *attrs):
        """シーケンスの各要素から複数の属性を選択し新しいフォームに射影します。

        **Parameters:**

        * **self** - `Query` オブジェクト自身またはIterable[T]
        * **\*attrs: str** - 選択する属性

        **Returns:** `Query` | `QueryDict[K, V]`

        Usage:
        ```
        >>> import pnq
        >>> pnq.query([str]).select_attrs("__class__", "__name__")
        [(<class 'type'>, 'str')]
        ```
        """
        pass

    def cast(self, type):
        """シーケンスの型注釈を変更します。この関数は、エディタの型解釈を助けるためだけに存在し、実行速度に影響を及ぼしません。
        実際に型を変更する場合は、`map`を使用してください。

        **Parameters:**

        * **self** - `Query` オブジェクト自身またはIterable[T]
        * **type: str** - 新しい型注釈

        **Returns:** `Query` | `QueryDict[K, V]`

        Usage:
        ```
        >>> import pnq
        >>> pnq.query([1]).cast(float)
        ```
        """
        pass

    def skip(self, count: int):
        pass

    def take(self, count: int):
        pass

    def range(self, from_: int, to: int = None):
        pass

    def pagenate(self, *, page: int, size: int):
        from_ = (page - 1) * size
        to = from_ + size
        self.range(from_, to)


"""
filter
map
select
select_attr
select_items
select_attrs
unpack
unpack_kw
cast
skip
take
pagenate
range
"""
