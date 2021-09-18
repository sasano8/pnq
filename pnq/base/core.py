from enum import Flag
from typing import TYPE_CHECKING

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
    """iter aiter両対応のイテラブルをラップするために使います（Queryクラスをチェインするのに使います）。"""

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


class QueryNormal(Query):
    """同期イテレータを両対応するために使います"""

    iter_type = IterType.BOTH

    def __init__(self, source):
        # sourceは__iter__しか実装していないので、
        # 自身を渡して__iter__と__aiter__を持っていると錯覚させる
        self.run_iter_type = IterType.BOTH
        super().__init__(self)
        self.source = source

    def _impl_iter(self):
        return self.source.__iter__()

    @final
    async def _impl_aiter(self):
        for v in self._impl_iter():
            yield v


class QuerySeq(QueryNormal):
    """リストなどをクエリ化します"""

    def _impl_iter(self):
        return self.source.__iter__()

    def __reversed__(self):
        return self.source.__reversed__()


class QueryDict(QueryNormal):
    """辞書などをクエリ化します"""

    def _impl_iter(self):
        return self.source.items().__iter__()

    def __reversed__(self):
        return self.source.items().__reversed__()


async def sync_to_async_iterator(it):
    for x in it:
        yield x


class QuerySyncToAsync(Query):
    """同期イテレータを非同期イテレータに変換します。
    もしくは、同期イテレータを取得できない場合、非同期イテレータの取得を試みます。"""

    iter_type = IterType.ASYNC

    def __init__(self, source):
        self.source = source
        self.run_iter_type = self.iter_type

    # def __iter__(self):
    #     if not (self.run_iter_type & IterType.NORMAL):
    #         raise NotImplementedError(f"{self.__class__} can't __iter__()")
    #     return self._impl_iter()

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


# class Map(Query):
#     def __init__(self, source, selector):
#         super().__init__(source)
#         self.selector = selector

#     def _impl_iter(self):
#         selector = self.selector
#         for x in self.source:
#             yield selector(x)

#     async def _impl_aiter(self):
#         selector = self.selector
#         async for x in self.source:
#             yield selector(x)


# class AsyncMap(Query):
#     iter_type = IterType.ASYNC

#     def __init__(self, source, func):
#         super().__init__(source)
#         self.func = func

#     async def _impl_aiter(self):
#         func = self.func
#         async for x in self.source:
#             yield await func(x)
