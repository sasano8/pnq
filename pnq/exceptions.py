class PnqException(Exception):
    """
    Pnqに関連する全ての例外の基底クラス
    """

    def __init__(self, msg: str = ""):
        super().__init__(msg)


# TODO: おそらくpnq in pnqでStopIterationが暴露した時に、想定外の事象が発生する
class StopIteration(PnqException):
    pass


class MustError(PnqException):
    """
    クエリがシーケンス全体あるいはシーケンスの要素に対して任意の特性を持っていることを要求したが失敗した

    関連: `must_unique`
    """


# KeyError IndexErrorを継承しても try: ... except KeyError としても補足できないっぽい
class NotFoundError(MustError, KeyError, IndexError):
    """
    クエリがキーに対応する要素を要求したが存在しない。
    IndexErrorとKeyErrorはKeyNotFoundErrorに置き換わります

    関連: `get` `must_get_many`
    """

    def __init__(self, key):
        super().__init__(f"{key}")
        self.key = key


class NoElementError(MustError):
    """
    クエリが何らかの要素を要求したが要素が存在しない

    関連: `one` `first` `last`
    """

    pass


class NotOneElementError(MustError):
    """
    クエリが要素がひとつであることを要求したが複数の要素が存在した

    関連: `one`
    """

    pass


class DuplicateElementError(MustError):
    """
    クエリが要素が重複していないことを要求したが重複を検知した

    関連: `must_unique`
    """
