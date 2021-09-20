from enum import Flag
from typing import TYPE_CHECKING, NoReturn

from . import finalizers

if TYPE_CHECKING:
    # python3.7には含まれていない
    from typing import final
else:
    final = lambda x: x  # noqa


class IterType(Flag):
    IMPOSSIBLE = 0
    NORMAL = 1
    ASYNC = 2
    BOTH = NORMAL | ASYNC


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


def to_query(source):
    if isinstance(source, list):
        return SyncSourceWrapper(source)
    elif hasattr(source, "__aiter__"):
        return Query(source)
    else:
        raise NotImplementedError()


class Query:
    """Queryクラスをチェインするのに使うか、__iter__と__aiter__の挙動をソースに任せる場合に使います。"""

    iter_type = IterType.BOTH

    def __init__(self, source):
        self.source = source
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

    def __iter__(self):
        if not (self.run_iter_type & IterType.NORMAL):
            raise NotImplementedError(f"{self.__class__} can't __iter__()")
        return self._impl_iter()

    def __aiter__(self):
        if not (self.run_iter_type & IterType.ASYNC):
            raise NotImplementedError(f"{self.__class__} can't __aiter__()")
        return self._impl_aiter()

    def _impl_iter(self):
        return self.source.__iter__()

    def _impl_aiter(self):
        return self.source.__aiter__()

    to = finalizers.to


class QueryNormal(Query):
    """同期イテレータを両対応するために使います"""

    iter_type = IterType.BOTH

    def __init__(self, source):
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


class QueryAsync(Query):
    """非同期イテレータのみ対応のクエリ"""

    iter_type = IterType.ASYNC

    def __init__(self, source):
        self.source = source
        self.run_iter_type = IterType.ASYNC

        if not hasattr(source, "__aiter__"):
            raise TypeError(f"{source} not has __aiter__")

    def _impl_iter(self):
        raise NotImplementedError()

    def _impl_aiter(self):
        return self.source.__aiter__()


class QuerySeq(QueryNormal):
    """リストなどをクエリ化します"""

    def _impl_iter(self):
        return self.source.__iter__()

    # def __reversed__(self):
    #     return self.source.__reversed__()


class QueryDict(QueryNormal):
    """辞書などをクエリ化します"""

    def _impl_iter(self):
        return self.source.items().__iter__()

    # def __reversed__(self):
    #     return self.source.items().__reversed__()


class QuerySet(QueryNormal):
    pass


async def sync_to_async_iterator(it):
    for x in it:
        yield x


class QuerySyncToAsync(Query):
    """同期イテレータを非同期イテレータに変換します。
    もしくは、同期イテレータを取得できない場合、非同期イテレータの取得を試みます。"""

    iter_type = IterType.ASYNC

    def __init__(self, source):
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
