from enum import Flag
from typing import (
    TYPE_CHECKING,
    AsyncIterable,
    AsyncIterator,
    Generic,
    Iterable,
    Iterator,
    Mapping,
    Tuple,
    TypeVar,
    Union,
)

from .protocols import IterType, PQuery

# from . import finalizers
# from pnq.protocols import WrappedQuery


if TYPE_CHECKING:
    # python3.7には含まれていない
    from typing import final
else:
    final = lambda x: x  # noqa

T = TypeVar("T")
K = TypeVar("K")
V = TypeVar("V")


def get_iter_type(source):
    run_iter_type = getattr(source, "run_iter_type", None)
    if run_iter_type:
        return run_iter_type
    elif hasattr(source, "__aiter__"):
        if hasattr(source, "__iter__"):
            return IterType.BOTH
        else:
            return IterType.ASYNC
    elif hasattr(source, "__iter__"):
        return IterType.NORMAL
    else:
        raise TypeError(f"{source} has no __iter__ or __aiter__")


def set_iter_type(self, source):
    source_iter_type = get_iter_type(source)

    # ソースの属性を継承し、クエリでタイプが強制された時はそのタイプを使う
    if self.iter_type == IterType.BOTH:
        self.run_iter_type = source_iter_type
    else:
        if self.iter_type == IterType.ASYNC:
            self.run_iter_type = self.iter_type

            if source_iter_type & IterType.ASYNC:
                pass
            else:
                # aiterのみ実行可能にする
                self.source = QuerySyncToAsync(self.source)
        else:
            raise TypeError("can not convert sync iterator to any iteraotr.")


class QueryNode(PQuery[T]):
    """イテレータに関する基本的な実装を持つクエリノード基底。

    WrappedQuery (再帰的内省) と iter_type 管理 (sync/async 振り分け) を統合した
    pnq の基底クラス。本クラスを継承するクラスは __iter__ / __aiter__ の挙動を
    source に委譲する (strict)。sync→async wrapping のようなブリッジ機能は派生
    クラス (queries.QueryRoot 等) が担当する。
    """

    iter_type = IterType.BOTH

    def __init__(self, source: Union[Iterable[T], AsyncIterable[T]]):
        self.source = source
        set_iter_type(self, source)

    def __iter__(self) -> Iterator[T]:
        if not (self.run_iter_type & IterType.NORMAL):
            raise NotImplementedError(
                f"{self.__class__.__name__}({self.source}) can't __iter__()"
            )
        return self._impl_iter()

    def __aiter__(self) -> AsyncIterator[T]:
        if not (self.run_iter_type & IterType.ASYNC):
            raise NotImplementedError(
                f"{self.__class__.__name__}({self.source}) can't __aiter__()"
            )
        return self._impl_aiter()

    def _impl_iter(self):
        return self.source.__iter__()

    def _impl_aiter(self):
        return self.source.__aiter__()


class QueryNormal(QueryNode[T]):
    """同期イテレータを両対応するために使います"""

    iter_type = IterType.BOTH

    def __init__(self, source: Iterable[T]):
        # super().__init__(self)
        self.source = source
        self.run_iter_type = IterType.BOTH

        if not hasattr(source, "__iter__"):
            raise TypeError(f"{source} not has __iter__")

    def _impl_iter(self):
        return self.source.__iter__()

    @final
    async def _impl_aiter(self):
        for v in self._impl_iter():
            yield v


class QueryAsync(QueryNode[T]):
    """非同期イテレータのみ対応のクエリ"""

    iter_type = IterType.ASYNC

    def __init__(self, source: Iterable[T]):
        self.source = source
        self.run_iter_type = IterType.ASYNC

        if not hasattr(source, "__aiter__"):
            raise TypeError(f"{source} not has __aiter__")

    def _impl_iter(self):
        raise NotImplementedError()

    def _impl_aiter(self):
        return self.source.__aiter__()


class PnqInternalSeq(QueryNormal[T]):
    """リストなどをクエリ化します"""

    def _impl_iter(self):
        return self.source.__iter__()

    # def __reversed__(self):
    #     return self.source.__reversed__()


class PnqInternalDict(QueryNormal[Tuple[K, V]]):
    """辞書などをクエリ化します"""

    if TYPE_CHECKING:

        def __init__(self, source: Mapping[K, V]): ...

    def _impl_iter(self):
        return self.source.items().__iter__()  # type: ignore

    # def __reversed__(self):
    #     return self.source.items().__reversed__()


class PnqInternalSet(QueryNormal[T]):
    pass


async def sync_to_async_iterator(it):
    for x in it:
        yield x


class QuerySyncToAsync(QueryNode[T]):
    """同期イテレータを非同期イテレータに変換します。
    もしくは、同期イテレータを取得できない場合、非同期イテレータの取得を試みます。"""

    iter_type = IterType.ASYNC

    def __init__(self, source: Iterable[T]):
        # super().__init__(self)
        self.source = source
        self.run_iter_type = self.iter_type

    def _impl_iter(self):
        raise NotImplementedError()

    def _impl_aiter(self):
        it = None
        try:
            it = iter(self.source)
        except Exception:
            pass

        if it:
            return sync_to_async_iterator(it)
        else:
            return self.source.__aiter__()
