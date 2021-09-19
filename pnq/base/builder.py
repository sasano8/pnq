from .core import Query, QueryAsync, QueryDict, QueryNormal, QuerySeq, QuerySet


class Builder:
    QUERY_BOTH = Query
    QUERY_ASYNC = QueryAsync
    QUERY_NORMAL = QueryNormal
    QUERY_SEQ = QuerySeq
    QUERY_DICT = QueryDict
    QUERY_SET = QuerySet

    @classmethod
    def query(cls, source):
        if isinstance(source, list):
            return cls.QUERY_SEQ(source)
        elif isinstance(source, dict):
            return cls.QUERY_DICT(source)
        elif isinstance(source, tuple):
            return cls.QUERY_SEQ(source)
        elif isinstance(source, set):
            return cls.QUERY_SET(source)
        elif isinstance(source, frozenset):
            return cls.QUERY_SET(source)
        elif isinstance(source, Query):
            return source
        else:
            has_iter = hasattr(source, "__iter__")
            has_aiter = hasattr(source, "__aiter__")

            if has_iter and has_aiter:
                return cls.QUERY_BOTH(source)
            elif has_iter:
                return cls.QUERY_NORMAL(source)
            elif has_aiter:
                return cls.QUERY_ASYNC(source)
            else:
                raise TypeError()
