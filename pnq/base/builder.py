from .core import Query, QueryDict, QueryNormal, QuerySeq


class Builder:
    QUERY_BOTH = Query
    QUERY_ASYNC = None
    QUERY_NORMAL = QueryNormal
    QUERY_SEQ = QuerySeq
    QUERY_DICT = QueryDict

    @classmethod
    def query(cls, source):
        if isinstance(source, Query):
            return source
        elif isinstance(source, list):
            return cls.QUERY_SEQ(source)
        elif isinstance(source, dict):
            return cls.QUERY_DICT(source)
        elif isinstance(source, tuple):
            return cls.QUERY_SEQ(source)
        else:
            if hasattr(source, "__iter__"):
                return cls.QUERY_BOTH(source)
            elif hasattr(source, "__aiter__"):
                return cls.QUERY_ASYNC(source)
            else:
                raise TypeError()
