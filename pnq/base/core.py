from enum import Flag


class IterType(Flag):
    IMPOSSIBLE = 0
    NORMAL = 1
    ASYNC = 2
    BOTH = NORMAL | ASYNC


def get_iter_type(source):
    if hasattr(source, "__aiter__"):
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
        self.source_iter_type = get_iter_type(source)

        # ソースの属性を継承し、クエリでタイプが強制された時はそのタイプを使う
        if self.iter_type == IterType.BOTH:
            self.run_iter_type = self.source_iter_type
        else:
            if self.iter_type == IterType.ASYNC:
                self.run_iter_type = self.iter_type

                if self.source_iter_type & IterType.ASYNC:
                    pass
                else:
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
        super().__init__(self)
        self.source = source

    def _impl_iter(self):
        return self.source.__iter__()

    async def _impl_aiter(self):
        for v in self.source.__iter__():
            yield v


class QuerySyncToAsync(Query):
    """同期イテレータを非同期イテレータに変換します。
    もしくは、同期イテレータを取得できない場合、非同期イテレータの取得を試みます。"""

    iter_type = IterType.ASYNC

    def __iter__(self):
        if not (self.run_iter_type & IterType.NORMAL):
            raise NotImplementedError(f"{self.__class__} can't __iter__()")
        return self._impl_iter()

    def _impl_iter(self):
        raise NotImplementedError()

    async def _impl_aiter(self):
        it = None
        try:
            it = iter(self.source)
        except Exception:
            pass

        if it:
            for v in it:
                yield v
        else:
            async for v in self.source:
                yield v


class Map(Query):
    def __init__(self, source, selector):
        super().__init__(source)
        self.selector = selector

    def _impl_iter(self):
        selector = self.selector
        for x in self.source:
            yield selector(x)

    async def _impl_aiter(self):
        selector = self.selector
        async for x in self.source:
            yield selector(x)


class AsyncMap(Query):
    iter_type = IterType.ASYNC

    def __init__(self, source, func):
        super().__init__(source)
        self.func = func

    async def _impl_aiter(self):
        func = self.func
        async for x in self.source:
            yield await func(x)
