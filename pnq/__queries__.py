# type: ignore

from functools import wraps
from operator import attrgetter, itemgetter
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    FrozenSet,
    Generic,
    Iterable,
    Iterator,
    List,
    Literal,
    Mapping,
    NoReturn,
    Sequence,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
    no_type_check,
    overload,
)

from . import actions
from .base.exceptions import NoElementError, NotFoundError, NotOneElementError
from .core import LazyIterate as _LazyIterate
from .core import LazyReference as _LazyReference
from .core import piter, undefined

# from .exceptions import NoElementError, NotFoundError, NotOneElementError
from .op import TH_ASSIGN_OP
from .requests import Response

if TYPE_CHECKING:
    from .base import queries

T = TypeVar("T")
K = TypeVar("K")
V = TypeVar("V")
K2 = TypeVar("K2")
V2 = TypeVar("V2")
R = TypeVar("R")


__all__ = ["Query", "PairQuery", "IndexQuery", "ListEx", "DictEx", "SetEx", "query", "undefined"]



def lazy_iterate(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        return LazyIterate(func, *args, **kwargs)

    return wrapper


def lazy_reference(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        return LazyReference(func, *args, **kwargs)

    return wrapper


{% for query in queries %}


    {% if not query.is_index %}


class {{query.cls}}:
    def len(self) -> int:
        return actions.len(self)

    def exists(self) -> bool:
        return actions.exists(self)

    def all(self, selector: Callable[[{{query.row}}], Any]=lambda x: x) -> bool:
        return actions.all(self, selector)

    def any(self, selector: Callable[[{{query.row}}], Any]=lambda x: x) -> bool:
        return actions.any(self, selector)

    def contains(self, value, selector: Callable[[{{query.row}}], Any]=lambda x: x) -> bool:
        return actions.contains(self, value, selector)

    @overload
    def min(self, *, default=NoReturn) -> Union[{{query.row}}, NoReturn]: ...

    @overload
    def min(self, selector: Callable[[{{query.row}}], R]=lambda x: x, default=NoReturn) -> R: ...
    def min(self, selector: Callable[[{{query.row}}], R]=lambda x: x, default=NoReturn) -> R:
        return actions.min(self, selector, default)

    @overload
    def max(self, *, default=NoReturn) -> Union[{{query.row}}, NoReturn]: ...
    @overload
    def max(self, selector: Callable[[{{query.row}}], R]=lambda x: x, default=NoReturn) -> R: ...
    def max(self, selector: Callable[[{{query.row}}], R]=lambda x: x, default=NoReturn) -> R:
        return actions.max(self, selector, default)

    @overload
    def sum(self) -> {{query.row}}: ...
    @overload
    def sum(self, selector: Callable[[{{query.row}}], R]=lambda x: x) -> R: ...
    def sum(self, selector: Callable[[{{query.row}}], R]=lambda x: x) -> R:
        return actions.sum(self, selector)

    @overload
    def average(self) -> {{query.row}}: ...
    @overload
    def average(self, selector: Callable[[{{query.row}}], R]=lambda x: x) -> R: ...
    def average(self, selector: Callable[[{{query.row}}], R]=lambda x: x) -> R:
        return actions.average(self, selector)

    def reduce(self, seed: T, op: Union[TH_ASSIGN_OP, Callable[[Any, Any], Any]] = "+=", selector=lambda x: x) -> T:
        return actions.reduce(self, seed, op, selector)

    def concat(self, selector=lambda x: x, delimiter: str = "") -> str:
        return actions.concat(self, selector, delimiter)

    {% if query.is_pair %}

    @overload
    def to(self, func: Type[Mapping[K, V]]) -> Mapping[K, V]:
        ...

    @overload
    def to(self, func: Callable[[Iterable[Tuple[K, V]]], R]) -> R:
        ...

    {% endif %}

    @overload
    def to(self, func: Type[Iterable[T]]) -> Iterable[T]:
        ...

    @overload
    def to(self, func: Callable[[Iterable[T]], R]) -> R:
        ...

    def to(self, func: Callable[[Iterable[T]], R]) -> R:
        return actions.to(self, func)

    def each(self, func: Callable = lambda x: x):
        return actions.each(self, func)

    def each_unpack(self, func: Callable = lambda x: x):
        return actions.each_unpack(self, func)

    async def each_async(self, func: Callable = lambda x: x):
        return await actions.each_async(self, func)

    async def each_async_unpack(self, func: Callable = lambda x: x):
        return await actions.each_async_unpack(self, func)

    def one(self) -> {{query.row}}:
        return actions.one(self)

    def one_or(self, default: R) -> Union[{{query.row}}, R]:
        return actions.one_or(self, default)

    def one_or_raise(self, exc: Union[str, Exception]) -> {{query.row}}:
        return actions.one_or_raise(self, exc)

    def first(self) -> {{query.row}}:
        return actions.first(self)

    def first_or(self, default: R) -> Union[{{query.row}}, R]:
        return actions.first_or(self, default)

    def first_or_raise(self, exc: Union[str, Exception]) -> {{query.row}}:
        return actions.first_or_raise(self, exc)

    def last(self) -> {{query.row}}:
        return actions.last(self)

    def last_or(self, default: R) -> Union[{{query.row}}, R]:
        return actions.last_or(self, default)

    def last_or_raise(self, exc: Union[str, Exception]) -> {{query.row}}:
        return actions.last_or_raise(self, exc)

    @overload
    def cast(self, type: Type[Tuple[K2, V2]]) -> "{{pair.name}}[K2, V2]":
        pass

    @overload
    def cast(self, type: Type[R]) -> "Query[R]":
        pass

    def cast(self, type: Type[R]) -> "Query[R]":
        return self

    # @lazy_iterate
    def enumerate(self, start: int = 0, step: int = 1) -> "{{pair.name}}[int, {{query.row}}]":
        return queries.Enumerate(self, start, step)
        # yield from (x * step for x in enumerate(self, start))

    @overload
    def map(self, selector: Callable[[{{query.row}}], Tuple[K2, V2]]) -> "{{pair.name}}[K2, V2]":
        pass

    @overload
    def map(self, selector: Callable[[{{query.row}}], R]) -> "{{sequence.name}}[R]":
        pass

    def map(self, selector):
        return queries.Map(self, selector)
        # if selector is None:
        #     raise TypeError("selector cannot be None")
        # return LazyIterate(actions.map, self, selector)

    {% if query.is_pair %}

    @overload
    def select(self, item: Literal[0]) -> "Query[{{query.K}}]":
        ...

    @overload
    def select(self, item: Literal[1]) -> "Query[{{query.V}}]":
        ...

    {% endif %}

    @overload
    def select(self, field) -> "Query[Any]":
        ...

    @overload
    def select(self, field1, field2, *fields) -> "PairQuery":
        ...

    @overload
    def select(self, field, *fields) -> "Query[Tuple]":
        ...

    def select(self, *fields, attr: bool = False) -> "Query[Any]":
        return queries.Select(self, *fields, attr=attr)
        # if attr:
        #     selector = attrgetter(*fields)
        # else:
        #     selector = itemgetter(*fields)
        # return LazyIterate(actions.map, self, selector)

    def select_as_tuple(self, *fields, attr: bool = False) -> "Query[Tuple]":
        return queries.SelectAsTuple(self, *fields, attr=attr)
        # if len(fields) > 1:
        #     if attr:
        #         selector = attrgetter(*fields)
        #     else:
        #         selector = itemgetter(*fields)
        # elif len(fields) == 1:
        #     field = fields[0]
        #     if attr:
        #         selector = lambda x: (getattr(x, field),)
        #     else:
        #         selector = lambda x: (x[field],)
        # else:
        #     selector = lambda x: ()
        # return LazyIterate(actions.map, self, selector)

    def select_as_dict(self, *fields, attr: bool = False, default=NoReturn) -> "Query[Dict]":
        return queries.SelectAsDict(self, *fields, attr=attr, default=default)
        # if attr:
        #     if default is NoReturn:
        #         selector = lambda x: {k: getattr(x, k) for k in fields} # noqa
        #     else:
        #         selector = lambda x: {k: getattr(x, k, default) for k in fields} # noqa
        # else:
        #     if default is NoReturn:
        #         selector = lambda x: {k: x[k] for k in fields} # noqa
        #     else:
        #         def get(obj, k):
        #             try:
        #                 return obj[k]
        #             except Exception:
        #                 return default
        #         selector = lambda x: {k: get(x, k) for k in fields} # noqa
        # return LazyIterate(actions.map, self, selector)

    @lazy_iterate
    def unpack_pos(self, selector: Callable[..., R]) -> "{{sequence.name}}[R]":
        return queries.UnpackPos(self, selector=selector)
        # return LazyIterate(actions.unpack_pos, self, selector)

    @lazy_iterate
    def unpack_kw(self, selector: Callable[..., R]) -> "{{sequence.name}}[R]":
        return queries.UnpackKw(self, selector=selector)
        # return LazyIterate(actions.unpack_kw, self, selector)

    def group_by(self, selector: Callable[[{{query.row}}], Tuple[K2, V2]] = lambda x: x) -> "{{pair.name}}[K2, List[V2]]":
        return queries.GroupBy(self, selector=selector)
        # return LazyIterate(actions.group_by, self, selector)

    def pivot_unstack(self, default=None) -> "PairQuery[Any, List]":
        return queries.PivotUnstack(self, default=default)
        # return LazyIterate(actions.pivot_unstack, self, default)

    def pivot_stack(self) -> "Query[Dict]":
        return queries.PivotStack(self)
        # return LazyIterate(actions.pivot_stack, self)

    def join(self, right, on: Callable[[Tuple[list, list]], Callable], select):
        [].join(
            [],
            lambda left, right: left.name == right.name,
            lambda left, right: (left.name, right.age)
        )

        table(User).join(
            Item,
            on=User.id == Item.id
        ).select(User.id, Item.id)

        pass

    def request(self, func, retry: int = None) -> "Query[Response]":
        return queries.Request(self, func, retry)
        # return LazyIterate(actions.request, self, func, retry)

    def request_async(self, func, retry: int = None, timeout=None) -> "Query[Response]":
        return queries.RequestAsync(self, func, retry=retry, timeout=None)
        # return LazyIterate(actions.request_async, self, func, retry)

    @lazy_iterate
    def zip(self):
        raise NotImplementedError()

    @lazy_iterate
    def filter(self, predicate: Callable[[{{query.row}}], bool]) -> "{{query.str}}":
        return queries.Filter(self, predicate)
        # yield from filter(predicate, self)  # type: ignore

    @lazy_iterate
    def filter_type(self, *types: Type[R]) -> "{{query.str}}":
        return queries.FilterType(self, *types)

        # def normalize(type):
        #     if type is None:
        #         return None.__class__
        #     else:
        #         return type

        # types = tuple((normalize(x) for x in types))
        # return filter(lambda x: isinstance(x, *types), self)

    @overload
    def filter_unique(self) -> "{{query.str}}":
        ...

    @overload
    def filter_unique(self, selector: Callable[[{{query.row}}], Tuple[K2, V2]]) -> "{{pair.name}}[K2, V2]":
        ...

    @overload
    def filter_unique(self, selector: Callable[[{{query.row}}], R]) -> "{{sequence.name}}[R]":
        ...

    def filter_unique(self, selector=None):
        return queries.FilterUnique(self, selector=selector)
        # return LazyIterate(actions.filter_unique, self, selector)

    # @lazy_iterate
    def distinct(self, selector: Callable[[{{query.row}}], Any]) -> "{{query.str}}":
        return queries.FilterUnique(self, selector=selector)
        # duplicate = set()
        # for elm in self:
        #     value = selector(elm)
        #     if not value in duplicate:
        #         duplicate.add(value)
        #         yield elm

    def must(self, predicate: Callable[[{{query.row}}], bool], msg: str="") -> "{{query.str}}":
        """要素の検証に失敗した時例外を発生させる。"""
        return queries.Must(self, predicate, msg)
        # return LazyIterate(actions.must, self, predicate, msg)

    def must_type(self, type, *types: Type) -> "{{query.str}}":
        """要素の検証に失敗した時例外を発生させる。"""
        return queries.MustType(self, type, *types)
        # return LazyIterate(actions.must_type, self, (type, *types))

    def must_unique(self, selector: Callable[[T], R] = None):
        return queries.MustUnique(self, selector=selector)
        # return LazyIterate(actions.must_unique, self, selector)

    # @lazy_iterate
    # def skip(self, count: int) -> "{{query.str}}":
    #     current = 0

    #     try:
    #         while current < count:
    #             next(self)
    #             current += 1
    #     except StopIteration:
    #         return

    #     for elm in self:
    #         yield elm

    def skip(self, count: int) -> "{{query.str}}":
        return queries.Skip(self, count=count)

    # @lazy_iterate
    # def take(self, count: int) -> "{{query.str}}":
    #     current = 0

    #     try:
    #         while current < count:
    #             yield next(self)
    #             current += 1
    #     except StopIteration:
    #         return

    def take(self, count: int) -> "{{query.str}}":
        return queries.Take(self, count=count)

    def take_range(self, start: int = 0, stop: int = None) -> "{{query.str}}":
        return queries.TakeRange(self, start=start, stop=stop)
        # return LazyIterate(actions.take_range, self, start=start, stop=stop)

    def take_page(self, page: int, size: int) -> "{{query.str}}":
        return queries.TakePage(self, page=page, size=size)
        # start, stop = actions.take_page_calc(page, size)
        # return LazyIterate(actions.take_range, self, start=start, stop=stop)

    def order_by(self, *fields, desc: bool = False, attr: bool = False) -> "{{query.str}}":
        # if not len(fields):
        #     selector = None
        # else:
        #     if attr:
        #         selector = attrgetter(*fields)
        #     else:
        #         selector = itemgetter(*fields)

        # return LazyIterate(actions.order_by, self, selector=selector, desc=desc)
        return queries.OrderBy(self, *fields, desc=desc, attr=attr)

    def order_by_map(self, selector=None, *, desc: bool = False) -> "{{query.str}}":
        return queries.OrderByMap(self, selector=selector, desc=desc)
        # return LazyIterate(actions.order_by, self, selector=selector, desc=desc)

    def order_by_reverse(self) -> "{{query.str}}":
        return queries.OrderByReverse(self)
        # return LazyReference(actions.order_by_reverse, self)

    def order_by_shuffle(self) -> "{{query.str}}":
        return queries.OrderByShuffle(self)
        # return LazyIterate(actions.order_by_shuffle, self)

    def sleep(self, seconds: float) -> "{{query.str}}":
        return queries.Sleep(self, seconds)
        # return LazyIterate(actions.sleep, self, seconds=seconds)

    def sleep_async(self, seconds: float) -> "{{query.str}}":
        return queries.Sleep(self, seconds)
        # return LazyIterate(actions.sleep_async, self, seconds=seconds)

    def debug(self, breakpoint=lambda x: x, printer=print) -> "{{query.str}}":
        return queries.Debug(self, breakpoint=breakpoint, printer=printer)
        # return LazyIterate(actions.debug, self, breakpoint=breakpoint, printer=printer)

    def debug_path(self, selector_sync=lambda x: -10, selector_async=lambda x: 10) -> "{{query.str}}":
        return queries.DebugPath(self, selector_sync, selector_async)

    # if index query
    {% else %}
class {{query.cls}}:

    def get_many(self, *keys) -> "{{query.str}}":
        raise NotImplementedError()

    def must_get_many(self, *keys) -> "{{query.str}}":
        raise NotImplementedError()

    @overload
    def get(self, key: {{query.K}}) -> {{query.V}}:
        ...

    @overload
    def get(self, key: {{query.K}}, default: R = NoReturn) -> Union[{{query.V}}, R]:
        ...

    def get(self, key: {{query.K}}, default=NoReturn) -> Any:
        return actions.get(self, key, default)

    def get_or(self, key: {{query.K}}, default: R) -> Union[{{query.V}}, R]:
        return actions.get_or(self, key, default)

    def get_or_raise(self, key: {{query.K}}, exc: Union[str, Exception]) -> {{query.V}}:
        return actions.get_or_raise(self, key, exc)

    {% endif %}
{% endfor %}



# 継承時は右側に基底クラスを指定し、左へ上書きしていくイメージ

class Lazy:
    def len(self) -> int:
        return len(list(self))

    def exists(self) -> bool:
        return len(list(self)) > 0


class LazyIterate(Lazy, Query[T], _LazyIterate):
    pass


class LazyReference(Lazy, IndexQuery[int, T], Query[T], _LazyReference):
    pass


class Instance:
    def len(self) -> int:
        return len(self)

    def exists(self) -> bool:
        return len(self) > 0


class ListEx(Instance, IndexQuery[int, T], Query[T], List[T]):
    def __piter__(self):
        return self.__iter__()

    # @lazy_reference
    # def reverse(self) -> "Query[T]":
    #     yield from reversed(self)

    @no_type_check
    def get_many(self, *keys):
        return LazyReference(actions.get_many_for_sequence, self, *keys)

    @no_type_check
    def must_get_many(self, *keys):
        return LazyReference(actions.must_get_many, self, *keys, typ="seq")


class TupleEx(Instance, IndexQuery[int, T], Query[T], Tuple[T]):
    def __piter__(self):
        return self.__iter__()

    # @lazy_reference
    # def reverse(self) -> "Query[T]":
    #     yield from reversed(self)

    @no_type_check
    def get_many(self, *keys):
        return LazyReference(actions.get_many_for_sequence, self, *keys)

    @no_type_check
    def must_get_many(self, *keys):
        return LazyReference(actions.must_get_many, self, *keys, typ="seq")


class DictEx(Instance, IndexQuery[K, V], PairQuery[K, V], Dict[K, V]):
    def __piter__(self):
        return self.items().__iter__()

    @lazy_reference
    def keys(self):
        yield from super().keys()

    @lazy_reference
    def values(self):
        yield from super().values()

    @lazy_reference
    def items(self):
        yield from super().items()

    # @lazy_reference
    # def reverse(self) -> "PairQuery[K, V]":
    #     for key in reversed(self):
    #         yield key, self[key]

    @no_type_check
    def get_many(self, *keys):
        return LazyReference(actions.get_many_for_mapping, self, *keys)

    @no_type_check
    def must_get_many(self, *keys):
        return LazyReference(actions.must_get_many, self, *keys, typ="map")


class SetEx(Instance, IndexQuery[T, T], Query[T], Set[T]):
    def __piter__(self):
        return self.__iter__()

    def __getitem__(self, key: T):
        if key in self:
            return key
        else:
            raise NotFoundError(key)

    def order_by_reverse(self) -> "Query[T]":
        raise NotImplementedError("Set has no order.")

    @no_type_check
    def get_many(self, *keys):
        return LazyReference(actions.get_many_for_set, self, *keys)

    @no_type_check
    def must_get_many(self, *keys):
        return LazyReference(actions.must_get_many, self, *keys, typ="set")


class FrozenSetEx(Instance, IndexQuery[T, T], Query[T], FrozenSet[T]):
    def __piter__(self):
        return self.__iter__()

    def __getitem__(self, key: T):
        if key in self:
            return key
        else:
            raise NotFoundError(key)

    def order_by_reverse(self) -> "Query[T]":
        raise NotImplementedError("Set has no order.")

    @no_type_check
    def get_many(self, *keys):
        return LazyReference(actions.get_many_for_set, self, *keys)

    @no_type_check
    def must_get_many(self, *keys):
        return LazyReference(actions.must_get_many, self, *keys, typ="set")



if TYPE_CHECKING:
    from .base import queries

else:
    import types

    class Queries:
        pass

    from .base import queries

    classess = Queries()

    for cls in queries.exports:
        baseclasses = (cls, Query[T])
        created = types.new_class(cls.__name__, baseclasses)

        setattr(classess, cls.__name__, created)

    queries = classess

from .base import builder


class QueryDict(queries.QueryDict):
    def get_many(self, *keys):
        return queries.FilterKeys(self, *keys)

    def get(self, key):
        try:
            return self.source[key]
        except KeyError:
            raise NotFoundError(key)

    def get_or(self, key, default=None):
        try:
            return self.get(key)
        except NotFoundError:
            return default

    def get_or_raise(self, key, exc: Union[str, Exception]):
        undefined = object()
        result = self.get_or(key, undefined)
        if result is undefined:
            if isinstance(exc, str):
                raise Exception(exc)
            else:
                raise exc
        else:
            return result

    # @no_type_check
    def must_get_many(self, *keys):
        return queries.MustKeys(self, *keys, typ="map")
        # return LazyReference(actions.must_get_many, self, *keys, typ="map")


class QuerySeq(queries.QuerySeq):
    def get_many(self, *keys):
        return queries.FilterKeys(self, *keys)

    def get(self, key):
        try:
            return self.source[key]
        except IndexError:
            raise NotFoundError(key)

    def get_or(self, key, default=None):
        try:
            return self.get(key)
        except NotFoundError:
            return default

    def get_or_raise(self, key, exc: Union[str, Exception]):
        undefined = object()
        result = self.get_or(key, undefined)
        if result is undefined:
            if isinstance(exc, str):
                raise Exception(exc)
            else:
                raise exc
        else:
            return result

    # @no_type_check
    def must_get_many(self, *keys):
        return queries.MustKeys(self, *keys, typ="seq")
        # return LazyReference(actions.must_get_many, self, *keys, typ="seq")

class QuerySet(queries.QuerySet):
    def order_by_reverse(self) -> "Query[T]":
        raise NotImplementedError("Set has no order.")

    def get_many(self, *keys):
        return queries.FilterKeys(self, *keys)

    def get(self, key):
        if key in self.source:
            return key
        else:
            raise NotFoundError(key)

    def get_or(self, key, default=None):
        try:
            return self.get(key)
        except NotFoundError:
            return default

    def get_or_raise(self, key, exc: Union[str, Exception]):
        undefined = object()
        result = self.get_or(key, undefined)
        if result is undefined:
            if isinstance(exc, str):
                raise Exception(exc)
            else:
                raise exc
        else:
            return result

    # @no_type_check
    def must_get_many(self, *keys):
        return queries.MustKeys(self, *keys, typ="set")
        # return LazyReference(actions.must_get_many, self, *keys, typ="set")


class QueryBuilder(builder.Builder):
    QUERY_BOTH = queries.Query
    QUERY_ASYNC = queries.QueryAsync
    QUERY_NORMAL = queries.QueryNormal
    QUERY_SEQ = QuerySeq
    QUERY_DICT = QueryDict
    QUERY_SET = QuerySet

@overload
def query(source: Mapping[K, V]) -> DictEx[K, V]:
    ...

@overload
def query(source: Iterable[T]) -> ListEx[T]:
    ...

@overload
def query(source) -> "Query[Any]":
    ...

def query(source):
    return QueryBuilder.query(source)

# def query(source):
#     if hasattr(source, "__piter__"):
#         return source
#     elif isinstance(source, dict):
#         return DictEx(source)
#     elif isinstance(source, list):
#         return ListEx(source)
#     elif isinstance(source, tuple):
#         return TupleEx(source)
#     elif isinstance(source, set):
#         return SetEx(source)
#     elif isinstance(source, frozenset):
#         return FrozenSetEx(source)
#     elif hasattr(source, "__iter__"):
#         return LazyIterate(iter, source)
#     else:
#         raise Exception()




