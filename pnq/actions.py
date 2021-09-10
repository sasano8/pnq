from typing import Mapping

from .exceptions import DuplicateError

__iter = iter
__next = next
__map = map
__filter = filter
__range = range
__all = all
__any = any
__max = max
__min = min
__sum = sum


marked = []


def mark(func):
    marked.append(func)
    return func


###########################################
# generator
###########################################
@mark
def iter(self):
    """イテラブルまたはマッピングからイテレータを取得します。
    マッピングの場合は、キーバリューのタプルを返すイテレータを取得します。

    Parameters:

    * self: イテレータを取得するイテラブルまたはマッピング

    Returns: 取得したイテレータを内包するクエリ

    Usage:
    ```
    >>> pnq.iter([1, 2]).to_list()
    [1, 2]
    >>> pnq.iter([{"id": 1, "name": "bob"}]).to_list()
    [("id", 1), ("name", "bob")]
    ```
    """
    it = getattr(self, "__piter__", None)
    if it:
        yield from it
    else:
        if isinstance(self, Mapping):
            yield from iter(self.items())
        else:
            yield from iter(self)


@mark
def infinite(func, *args, **kwargs):
    """渡した関数を無限に実行するイテレータを生成します。
    無限に繰り返されるため、`take`等で終了条件を設定するように注意してください。

    Parameters:

    * func(args, kwargs): 無限に実行する関数
    * args: 関数に渡す位置引数
    * kwargs: 関数に渡すキーワード引数

    Returns: 取得したイテレータを内包するクエリ

    Usage:
    ```
    >>> pnq.infinite(datetime.now).take(1).to_list()
    [datetime.datetime(2021, 9, 10, 3, 57, 54, 402467)]
    >>> pnq.infinite(datetime, 2020, 1, day=2).take(1).to_list()
    [datetime.datetime(2010, 1, 2, 0, 0)]
    ```
    """
    while True:
        yield func(*args, **kwargs)


@mark
def repeat(value):
    """同じ値を繰り返す無限イテレータを生成します。

    Parameters:

    * value: 繰り返す値

    Returns: 取得したイテレータを内包するクエリ

    Usage:
    ```
    >>> pnq.repeat(5).take(3).to_list()
    [5, 5, 5]
    ```
    """
    from itertools import repeat

    yield from repeat(value)


@mark
def count(start=0, step=1):
    """連続した値を無限に返すイテレータを生成します。
    無限に繰り返されるため、`take`等で終了条件を設定するように注意してください。

    Parameters:

    * start: 開始値
    * step: 増分

    Returns: 取得したイテレータを内包するクエリ

    Usage:
    ```
    >>> pnq.count().take(3).to_list()
    [0, 1, 2]
    >>> pnq.count(1, 2).take(3).to_list()
    [1, 3, 5]
    ```
    """
    from itertools import count

    yield from count(start, step)


@mark
def cycle(iterable, repeat=None):
    """イテラブルが返す値を無限に繰り返すイテレータを生成します。

    Parameters:

    * iterable: 繰り返すイテラブル
    * repeat: 繰り返す回数。Noneの場合は無限に繰り返します。

    Returns: 取得したイテレータを内包するクエリ

    Usage:
    ```
    >>> pnq.cycle([1,2,3]).take(4).to_list()
    [1, 2, 3, 1]
    >>> pnq.cycle([1,2,3], repeat=2).to_list()
    [1, 2, 3, 1, 2, 3]
    ```
    """
    from itertools import cycle

    yield from cycle(iterable, repeat)


@mark
def range(*args, **kwargs):
    """指定した開始数と終了数までの連続した値を返すイテレータを生成します。

    Parameters:

    * stop: 終了数

    Parameters:

    * start: 開始数
    * stop: 終了数
    * step: 増分

    Returns: 取得したイテレータを内包するクエリ

    Usage:
    ```
    >>> pnq.range(5).to(list)
    [0, 1, 2, 3, 4]
    >>> pnq.range(0, 3, 2).to(list)
    [0, 2, 4]
    ```
    """
    yield from __range(*args, **kwargs)


###########################################
# mapping
###########################################
@mark
def map(self, selector):
    """シーケンスの各要素を新しいフォームに射影します。
    str関数を渡した場合、利便性のため`None`は`""`を返します（Pythonの標準動作は`"None"`を返します）。

    Parameters:

    * self: 変換対象のシーケンス
    * selector(x): 各要素に対する変換関数

    Returns: 変換関数で得られた要素を含むクエリ

    Usage:
    ```
    >>> pnq.query([1]).map(lambda x: x * 2).to_list()
    [2]
    >>> pnq.query([None]).map(str).to_list()
    [""]
    ```
    """
    return __map(selector, self)


@mark
def unpack(self, selector):
    """unpack_posかunpack_kwの別名にする予定です。"""
    pass


@mark
def unpack_pos(self, selector):
    """シーケンスの各要素をアンパックし、新しいフォームに射影します。

    Parameters:

    * self: 変換対象のシーケンス
    * selector(*args): 各要素に対する変換関数

    Returns: 変換関数で得られた要素を含むクエリ

    Usage:
    ```
    >>> pnq.query([(1, 2)]).unpack_pos(lambda arg1, arg2: arg1)).to_list()
    [1]
    >>> pnq.query([(1, 2, 3, 4, 5)]).unpack_pos(lambda arg1, arg2, *args: args).to_list()
    [(3, 4, 5)]
    ```
    """
    pass


@mark
def unpack_kw(self, selector):
    """シーケンスの各要素をキーワードアンパックし、新しいフォームに射影します。

    Parameters:

    * self: 変換対象のシーケンス
    * selector(kwargs): 各要素に対する変換関数

    Returns: 変換関数で得られた要素を含むクエリ

    Usage:
    ```
    >>> pnq.query([{"id": 1, "name": "bob"}]).unpack_kw(lambda id, name: name)).to_list()
    ["bob"]
    >>> pnq.query([{"id": 1, "name": "bob", "age": 20}]).unpack_kw(lambda id, name, **kwargs: kwargs)).to_list()
    [{"age": 20}]
    ```
    """
    pass


@mark
def select(self, item, *items):
    """シーケンスの各要素からアイテムを選択し新しいフォームに射影します。
    複数のアイテムを選択した場合は、タプルとして射影します。
    `select_item`の別名です。

    Parameters:

    * self: 変換対象のシーケンス
    * item: 各要素から選択するアイテム
    * items: 各要素から選択するアイテム

    Returns: 選択したアイテムまたは複数のアイテム（タプル）を含むクエリ

    Usage:
    ```
    >>> pnq.query([(1, 2)]).select(0).to_list()
    [1]
    >>> pnq.query([{"id": 1, "name": "a"}]).select("id", "name").to_list()
    [(1, "a")]
    ```
    """
    pass


@mark
def select_as_dict(self, *fields, attr: bool = False):
    """シーケンスの各要素からアイテムまたは属性を選択し辞書として新しいフォームに射影します。

    Parameters:

    * self: 変換対象のシーケンス
    * fields: 選択するアイテムまたは属性
    * attr: 属性から取得する場合はTrueとする

    Returns: 選択したアイテムを含む辞書を含むクエリ

    Usage:
    ```
    >>> pnq.query([(1, 2)]).select_as_dict(0).to_list()
    [{0: 1}]
    >>> pnq.query([{"id": 1, "name": "a", "age": 20}]).select_as_dict("id", "name").to_list()
    [{"id": 1, "name": "a"}]
    ```
    """
    pass


@mark
def select_as_tuple(self, *fields, attr: bool = False):
    pass


@mark
def select_item(self, item, *items):
    """`select`を参照ください。"""
    pass


@mark
def select_attr(self, attr, *attrs):
    """シーケンスの各要素から属性を選択し新しいフォームに射影します。
    複数の属性を選択した場合は、タプルとして射影します。

    Parameters:

    * self: 変換対象のシーケンス
    * attr: 各要素から選択する属性
    * attrs: 各要素から選択する属性

    Returns: 選択した属性または複数の属性（タプル）を含むクエリ

    Usage:
    ```
    >>> obj = Person(id=1, name="bob")
    >>> pnq.query([obj]).select_attr("name").to_list()
    ["bob"]
    >>> pnq.query([obj]).select_attr("id", "name").to_list()
    [(1, "bob")]
    ```
    """
    pass


@mark
def select_items(self, *items):
    """シーケンスの各要素から複数のアイテムを選択し新しいフォームに射影します。
    `select_item`と異なり、常にタプルを返します。

    Parameters:

    * self: 変換対象のシーケンス
    * items: 各要素から選択するアイテム

    Returns: 複数のアイテム（タプル）を含むクエリ

    Usage:
    ```
    >>> pnq.query([(1, 2)]).select_items(0).to_list()
    [(1,)]
    >>> pnq.query([{"id": 1, "name": "a"}]).select_items("id", "name").to_list()
    [(1, "a")]
    ```
    """
    pass


@mark
def select_attrs(self, *attrs):
    """シーケンスの各要素から複数の属性を選択し新しいフォームに射影します。
    `select_attr`と異なり、常にタプルを返します。

    Parameters:

    * self: 変換対象のシーケンス
    * attrs: 各要素から選択する属性

    Returns: 複数の属性（タプル）を含むクエリ

    Usage:
    ```
    >>> obj = Person(id=1, name="bob")
    >>> pnq.query([obj]).select_attrs("name").to_list()
    [("bob",)]
    >>> pnq.query([obj]).select_attrs("id", "name").to_list()
    [(1, "bob")]
    ```
    """
    pass


@mark
def cast(self, type):
    """シーケンスの型注釈を変更します。この関数はエディタの型解釈を助けるためだけに存在し、何も処理を行いません。

    実際に型を変更する場合は、`map`を使用してください。

    Parameters:

    * self: 変換対象のシーケンス
    * type: 新しい型注釈

    Returns: 参照したシーケンスをそのまま返す

    Usage:
    ```
    >>> pnq.query([1]).cast(float)
    ```
    """
    pass


@mark
def enumerate(self, start: int = 0, step: int = 1):
    """シーケンスの各要素とインデックスを新しいフォームに射影します。

    Parameters:

    * self: 変換対象のシーケンス
    * start: 開始インデックス
    * step: 増分

    Returns: インデックスと要素（タプル）を含むクエリ

    Usage:
    ```
    >>> pnq.query([1, 2]).enumrate().to_list()
    [(0, 1), (1, 2)]
    >>> pnq.query([1, 2]).enumrate(5).to_list()
    [(5, 1), (6, 2)]
    >>> pnq.query([1, 2]).enumrate(0, 10)).to_list()
    [(0, 1), (10, 2)]
    ```
    """
    pass


@mark
def group_by(self, selector):
    """シーケンスの各要素からセレクタ関数でキーとバリューを取得し、キーでグループ化されたシーケンスを生成します。
    セレクタ関数を指定しない場合、各要素がすでにキーバリューのタプルであることを期待し、キーでグループ化します。

    Parameters:

    * self: 変換対象のシーケンス
    * selector: キーとバリューを選択する関数

    Returns: キーと要素（タプル）を含むクエリ

    Usage:
    ```
    >>> data = [
    >>>   {"name": "banana", "color": "yellow", "count": 3},
    >>>   {"name": "apple", "color": "red", "count": 2},
    >>>   {"name": "strawberry", "color": "red", "count": 5},
    >>> ]
    >>> pnq.query(data).group_by(lambda x: x["color"], x["name"]).to_list()
    [("yellow": ["banana"]), ("red": ["apple", "strawberry"])]
    >>> pnq.query(data).select("color", "count").group_by().to_dict()
    {"yellow": [3], "red": [2, 5]}
    ```
    """


@mark
def join(self, right, on, select):
    pass


###########################################
# Expander set operation
###########################################

# https://docs.aws.amazon.com/ja_jp/redshift/latest/dg/r_UNION.html
@mark
def union_intersect(self):
    """共通部分のみ抽出"""
    pass


@mark
def union(self, *iterables):
    """全ての行を集合に含み重複は許可しない"""
    pass


@mark
def union_all(self):
    """全ての行を集合に含む"""
    pass


@mark
def union_minus(self):
    """1つ目の問い合わせには存在するが、2つ目の問い合わせには存在しないデータを抽出
    exceptと同じ意味
    """
    pass


@mark
def zip(self, *iterables):
    pass


###########################################
# filtering
###########################################
@mark
def filter(self, predicate):
    """述語に基づいてシーケンスの要素をフィルタ処理します。

    Parameters:

    * self: フィルタ対象のシーケンス
    * predicate: 条件を満たすか検証する関数

    Returns: 条件を満たす要素を含むクエリ

    Usage:
    ```
    >>> pnq.query([1, 2]).filter(lambda x: x == 1).to_list()
    [1]
    >>> pnq.query({1: True, 2: False, 3: True}).filter(lambda x: x[1] == True).to_list()
    [(1, True), (3, True)]
    ```
    """
    return __filter(predicate, self)


@mark
def must(self, predicate):
    """述語に基づいてシーケンスの要素を検証します。
    検証に失敗した場合、即時に例外が発生します。

    Parameters:

    * self: フィルタ対象のシーケンス
    * predicate: 条件を満たすか検証する関数

    Returns: 条件を満たす要素を含むクエリ

    Usage:
    ```
    >>> pnq.query([1, 2]).must(lambda x: x == 1).to_list()
    raise ValueError("2")
    >>> pnq.query({1: True, 2: False, 3: True}).must(lambda x: x[1] == True).to_list()
    raise ValueError("(2, False)")
    ```
    """
    for elm in self:
        if not predicate(elm):
            raise ValueError(elm)
        yield elm


@mark
def filter_type(self, *types):
    for elm in self:
        if isinstance(elm, *types):
            yield elm


@mark
def must_type(self, *types):
    for elm in self:
        if not isinstance(elm, *types):
            raise TypeError(f"{elm} is not {types}")
        yield elm


@mark
def filter_unique(self, selector):
    duplidate = set()
    for elm in self:
        value = selector(elm)
        if value in duplidate:
            pass
        else:
            duplidate.add(value)
            yield elm


@mark
def must_unique(self, selector):
    duplidate = set()
    for elm in self:
        value = selector(elm)
        if value in duplidate:
            raise DuplicateError(elm)
        else:
            duplidate.add(value)
            yield elm


@mark
def filter_items(self, *items):
    for elm in self:
        if all(elm[x] for x in items):
            yield elm


@mark
def must_items(self, *items):
    for elm in self:
        if all(elm[x] for x in items):
            yield elm
        else:
            raise ValueError(elm)


@mark
def filter_attrs(self, *attrs):
    for elm in self:
        if all(getattr(elm, x) for x in attrs):
            yield elm


@mark
def must_attrs(self, *attrs):
    for elm in self:
        if all(getattr(elm, x) for x in attrs):
            yield elm
        else:
            raise ValueError(elm)


@mark
def get_many(self, *keys):
    ...


###########################################
# partitioning
###########################################

###########################################
# aggregating
###########################################
@mark
def all(self):
    pass


###########################################
# order
###########################################
@mark
def reverse(self):
    yield from reversed(list(iter(self)))


###########################################
# finalizer
###########################################
@mark
def dispatch(self, func):
    for elm in self:
        func(elm)


@mark
def exec(self, func, unpack_kw: bool = False):
    """シーケンスから流れてくる値を関数に送出するように要求します。
    例外はキャッチされ、実行結果を返すイテレータを生成します。
    関数はイテレーションを要求されるまで実行されません。

    Parameters:

    * self: フィルタ対象のシーケンス
    * func: 値の送出先の関数
    * unpack_kw: 値をキーワードアンパックする

    Returns: 実行結果を含むタプル

    Usage:
    ```
    >>> def do_something(val):
    >>>   if val:
    >>>     return 1
    >>>   else:
    >>>     raise ValueError(val)
    >>>
    >>> for elm, err, result in pnq.query([True, False]).exec(do_something):
    >>>   if not err:
    >>>     print(elm, err, result)
    True, None, 1
    >>>   else:
    >>>     print(elm, err, result)
    False, ValueError("False"), None
    ```
    """
    for elm in self:
        err = None
        result = None
        try:
            result = func(elm)
        except Exception as err:  # noqa
            pass

        yield elm, err, result
