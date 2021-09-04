class QueryException(Exception):
    pass


# TODO: おそらくpnq in pnqでStopIterationが暴露した時に、想定外の事象が発生する
class StopIteration(QueryException):
    pass


class NoElementError(QueryException):
    pass


class NotOneError(QueryException):
    pass
