from decimal import Decimal
from decimal import InvalidOperation as DecimalInvalidOperation
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    Mapping,
    NoReturn,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    overload,
)

if TYPE_CHECKING:
    from _typeshed import SupportsKeysAndGetItem

from pnq import actions

from .exceptions import (
    DuplicateElementError,
    NoElementError,
    NotFoundError,
    NotOneElementError,
)
from .op import MAP_ASSIGN_OP, TH_ASSIGN_OP, TH_ROUND

T = TypeVar("T")
K = TypeVar("K")
V = TypeVar("V")


# TODO: 組み込みの名前を返るのは頭が混乱するのでやめるべき
# TODO: やるなら一度別名で定義して、別モジュールに別名でインポートするべき


class EmptyLambda:
    def __call__(self, x):
        return x

    def __str__(self):
        return "lambda x: x"


lambda_empty = EmptyLambda()
# lambda_empty = lambda x: x
# lambda_empty.__str__ = lambda: "lambda x: x"


marked = []


def mark(func):
    marked.append(func)
    return func


def name_as(name):
    def wrapper(func):
        func.__name__ = name
        return func

    return wrapper


@name_as("list")
def __list(iterable: Iterable[T]) -> List[T]:
    return list(iterable)


@overload
def __dict(mapping: "SupportsKeysAndGetItem[K, V]") -> Dict[K, V]:
    return Dict[K, V](mapping)


@overload
def __dict(iterable: Iterable[Tuple[K, V]]) -> Dict[K, V]:
    return Dict[K, V](iterable)


@name_as("dict")
def __dict(iterable):
    return Dict[K, V](iterable)


###########################################
# generator
###########################################
@mark
@name_as("iter")
def __iter(self, selector=None):
    """イテラブルまたはマッピングからイテレータを取得します。
    マッピングの場合は、キーバリューのタプルを返すイテレータを取得します。

    Args:

    * self: イテレータを取得するイテラブルまたはマッピング
    * selector: 取得したイテレータから任意の要素を抽出する関数

    Returns: 取得したイテレータを内包するクエリ

    Usage:
    ```
    >>> pnq.iter([1, 2]).to(list)
    [1, 2]
    >>> pnq.iter([{"id": 1, "name": "bob"}]).to(list)
    [("id", 1), ("name", "bob")]
    >>> pnq.iter([{"id": 1, "name": "bob"}], lambda x: x[1]).to(list)
    [1, "bob"]
    ```
    """

    it = getattr(self, "__piter__", None)
    if it:
        it = it()
    else:
        if isinstance(self, Mapping):
            it = self.items()
        else:
            it = self

    return __get_iterator(it, selector)


def __get_iterator(iterator, selector):
    if selector is None:
        return iterator
    else:
        return map(selector, iterator)


@mark
def __next(self):
    raise NotImplementedError()


@mark
def value(*args, **kwargs):
    """１つの要素を返すイテレータを生成します。

    Args:

    * value: 返す値

    Returns: 取得したイテレータを内包するクエリ

    Usage:
    ```
    >>> pnq.value(1).to(list)
    [1]
    >>> pnq.value(name="test").to(list)
    [{"naem": "test"}]
    ```
    """
    if args and kwargs:
        raise ValueError("value() can't accept both positional and keyword arguments")

    if kwargs:
        val = kwargs

    elif len(args) > 1:
        val = args

    elif len(args) == 1:
        val = args[0]

    elif len(args) == 0:
        val = None

    else:
        raise NotImplementedError()

    yield val


@mark
def infinite(func, *args, **kwargs):
    """渡した関数を無限に実行するイテレータを生成します。
    無限に繰り返されるため、`take`等で終了条件を設定するように注意してください。

    Args:

    * func(args, kwargs): 無限に実行する関数
    * args: 関数に渡す位置引数
    * kwargs: 関数に渡すキーワード引数

    Returns: 取得したイテレータを内包するクエリ

    Usage:
    ```
    >>> pnq.infinite(datetime.now).take(1).to(list)
    [datetime.datetime(2021, 9, 10, 3, 57, 54, 402467)]
    >>> pnq.infinite(datetime, 2020, 1, day=2).take(1).to(list)
    [datetime.datetime(2010, 1, 2, 0, 0)]
    ```
    """
    while True:
        yield func(*args, **kwargs)


@mark
def repeat(value):
    """同じ値を繰り返す無限イテレータを生成します。

    Args:

    * value: 繰り返す値

    Returns: 取得したイテレータを内包するクエリ

    Usage:
    ```
    >>> pnq.repeat(5).take(3).to(list)
    [5, 5, 5]
    ```
    """
    from itertools import repeat

    yield from repeat(value)


@mark
def count(start=0, step=1):
    """連続した値を無限に返すイテレータを生成します。
    無限に繰り返されるため、`take`等で終了条件を設定するように注意してください。

    Args:

    * start: 開始値
    * step: 増分

    Returns: 取得したイテレータを内包するクエリ

    Usage:
    ```
    >>> pnq.count().take(3).to(list)
    [0, 1, 2]
    >>> pnq.count(1, 2).take(3).to(list)
    [1, 3, 5]
    ```
    """
    from itertools import count

    yield from count(start, step)


@mark
def cycle(iterable, repeat=None):
    """イテラブルが返す値を無限に繰り返すイテレータを生成します。

    Args:

    * iterable: 繰り返すイテラブル
    * repeat: 繰り返す回数。Noneの場合は無限に繰り返します。

    Returns: 取得したイテレータを内包するクエリ

    Usage:
    ```
    >>> pnq.cycle([1,2,3]).take(4).to(list)
    [1, 2, 3, 1]
    >>> pnq.cycle([1,2,3], repeat=2).to(list)
    [1, 2, 3, 1, 2, 3]
    ```
    """
    from itertools import cycle

    yield from cycle(iterable, repeat)


@mark
@name_as("range")
def __range(*args, **kwargs):
    """指定した開始数と終了数までの連続した値を返すイテレータを生成します。

    Args:

    * stop: 終了数

    Args:

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
    yield from range(*args, **kwargs)


###########################################
# mapping
###########################################
@mark
@name_as("map")
def __map(self, selector):
    """シーケンスの各要素を新しいフォームに射影します。
    str関数を渡した場合、利便性のため`None`は`""`を返します（Pythonの標準動作は`"None"`を返します）。

    Args:

    * self: 変換対象のシーケンス
    * selector(x): 各要素に対する変換関数

    Returns: 変換関数で得られた要素を含むクエリ

    Usage:
    ```
    >>> pnq.query([1]).map(lambda x: x * 2).to(list)
    [2]
    >>> pnq.query([None]).map(str).to(list)
    [""]
    ```
    """
    return map(selector, self)


def starmap(self):
    from itertools import starmap

    # starmap(pow, [(2,5), (3,2), (10,3)]) --> 32 9 1000
    for args in self:
        yield function(*args)


def itertools__():
    from itertools import (
        chain,
        combinations,
        combinations_with_replacement,
        count,
        cycle,
        dropwhile,
        groupby,
        islice,
        permutations,
        product,
        repeat,
        starmap,
        takewhile,
        tee,
        zip_fillvalue,
        zip_longest,
    )


@mark
def map_flat(self, selector):
    pass


select_many = map_flat
flat_map = map_flat
"""
# product（デカルト積）
from itertools import product
mark=['♠', '♥', '♦', '♣']
suu = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
tramp = [m for m in product(mark,suu)]
print(tramp)
# [('♠', 'A'), ('♠', '2'), ...]
"""


@mark
def select_single_node(
    self,
    node_selector=lambda x: x.node,
    return_selector=lambda nest, parent, child: (nest, parent, child),
):
    pass


@mark
def map_recursive(self, selector):
    "nodeを再帰的に取得する"
    pass


@mark
def unpack(self, selector):
    """unpack_posかunpack_kwの別名にする予定です。"""
    pass


@mark
def unpack_pos(self, selector):
    """シーケンスの各要素をアンパックし、新しいフォームに射影します。

    Args:

    * self: 変換対象のシーケンス
    * selector(*args): 各要素に対する変換関数

    Returns: 変換関数で得られた要素を返すクエリ

    Usage:
    ```
    >>> pnq.query([(1, 2)]).unpack_pos(lambda arg1, arg2: arg1)).to(list)
    [1]
    >>> pnq.query([(1, 2, 3, 4, 5)]).unpack_pos(lambda arg1, arg2, *args: args).to(list)
    [(3, 4, 5)]
    ```
    """
    pass


@mark
def unpack_kw(self, selector):
    """シーケンスの各要素をキーワードアンパックし、新しいフォームに射影します。

    Args:

    * self: 変換対象のシーケンス
    * selector(kwargs): 各要素に対する変換関数

    Returns: 変換関数で得られた要素を返すクエリ

    Usage:
    ```
    >>> pnq.query([{"id": 1, "name": "bob"}]).unpack_kw(lambda id, name: name)).to(list)
    ["bob"]
    >>> pnq.query([{"id": 1, "name": "bob", "age": 20}]).unpack_kw(lambda id, name, **kwargs: kwargs)).to(list)
    [{"age": 20}]
    ```
    """
    pass


@mark
def select(self, item, *items):
    """シーケンスの各要素からアイテムを選択し新しいフォームに射影します。
    複数のアイテムを選択した場合は、タプルとして射影します。
    `select_item`の別名です。

    Args:

    * self: 変換対象のシーケンス
    * item: 各要素から選択するアイテム
    * items: 各要素から選択するアイテム

    Returns: 選択したアイテムまたは複数のアイテム（タプル）を返すクエリ

    Usage:
    ```
    >>> pnq.query([(1, 2)]).select(0).to(list)
    [1]
    >>> pnq.query([{"id": 1, "name": "a"}]).select("id", "name").to(list)
    [(1, "a")]
    >>> id, name = pnq.query([{"id": 1, "name": "a"}]).select("id", "name")
    ```
    """
    pass


@mark
def select_as_dict(self, *fields, attr: bool = False):
    """シーケンスの各要素からアイテムまたは属性を選択し辞書として新しいフォームに射影します。

    Args:

    * self: 変換対象のシーケンス
    * fields: 選択するアイテムまたは属性
    * attr: 属性から取得する場合はTrueとする

    Returns: 選択したアイテムを含む辞書を返すクエリ

    Usage:
    ```
    >>> pnq.query([(1, 2)]).select_as_dict(0).to(list)
    [{0: 1}]
    >>> pnq.query([{"id": 1, "name": "a", "age": 20}]).select_as_dict("id", "name").to(list)
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

    Args:

    * self: 変換対象のシーケンス
    * attr: 各要素から選択する属性
    * attrs: 各要素から選択する属性

    Returns: 選択した属性または複数の属性（タプル）を返すクエリ

    Usage:
    ```
    >>> obj = Person(id=1, name="bob")
    >>> pnq.query([obj]).select_attr("name").to(list)
    ["bob"]
    >>> pnq.query([obj]).select_attr("id", "name").to(list)
    [(1, "bob")]
    ```
    """
    pass


@mark
def select_items(self, *items):
    """シーケンスの各要素から複数のアイテムを選択し新しいフォームに射影します。
    `select_item`と異なり、常にタプルを返します。

    Args:

    * self: 変換対象のシーケンス
    * items: 各要素から選択するアイテム

    Returns: 複数のアイテム（タプル）を返すクエリ

    Usage:
    ```
    >>> pnq.query([(1, 2)]).select_items(0).to(list)
    [(1,)]
    >>> pnq.query([{"id": 1, "name": "a"}]).select_items("id", "name").to(list)
    [(1, "a")]
    ```
    """
    pass


@mark
def select_attrs(self, *attrs):
    """シーケンスの各要素から複数の属性を選択し新しいフォームに射影します。
    `select_attr`と異なり、常にタプルを返します。

    Args:

    * self: 変換対象のシーケンス
    * attrs: 各要素から選択する属性

    Returns: 複数の属性（タプル）を返すクエリ

    Usage:
    ```
    >>> obj = Person(id=1, name="bob")
    >>> pnq.query([obj]).select_attrs("name").to(list)
    [("bob",)]
    >>> pnq.query([obj]).select_attrs("id", "name").to(list)
    [(1, "bob")]
    ```
    """
    pass


@mark
def cast(self, type):
    """シーケンスの型注釈を変更します。この関数はエディタの型解釈を助けるためだけに存在し、何も処理を行いません。

    実際に型を変更する場合は、`map`を使用してください。

    Args:

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
@name_as("enumerate")
def __enumerate(self, start: int = 0, step: int = 1):
    """シーケンスの各要素とインデックスを新しいフォームに射影します。

    Args:

    * self: 変換対象のシーケンス
    * start: 開始インデックス
    * step: 増分

    Returns: インデックスと要素（タプル）を返すクエリ

    Usage:
    ```
    >>> pnq.query([1, 2]).enumerate().to(list)
    [(0, 1), (1, 2)]
    >>> pnq.query([1, 2]).enumerate(5).to(list)
    [(5, 1), (6, 2)]
    >>> pnq.query([1, 2]).enumerate(0, 10)).to(list)
    [(0, 1), (10, 2)]
    ```
    """
    pass


@mark
def group_by(self, selector):
    """シーケンスの各要素からセレクタ関数でキーとバリューを取得し、キーでグループ化されたシーケンスを生成します。
    セレクタ関数を指定しない場合、各要素がすでにキーバリューのタプルであることを期待し、キーでグループ化します。

    Args:

    * self: 変換対象のシーケンス
    * selector: キーとバリューを選択する関数

    Returns: キーと要素（タプル）を返すクエリ

    Usage:
    ```
    >>> data = [
    >>>   {"name": "banana", "color": "yellow", "count": 3},
    >>>   {"name": "apple", "color": "red", "count": 2},
    >>>   {"name": "strawberry", "color": "red", "count": 5},
    >>> ]
    >>> pnq.query(data).group_by(lambda x: x["color"], x["name"]).to(list)
    [("yellow": ["banana"]), ("red": ["apple", "strawberry"])]
    >>> pnq.query(data).select("color", "count").group_by().to_dict()
    {"yellow": [3], "red": [2, 5]}
    ```
    """


@mark
def join(self, right, on, select):
    pass


@mark
def group_join(self, right, on, select):
    pass


@mark
def request(self, func, unpack: bool = True, timeout: float = None, retry: int = None):
    """シーケンスから流れてくる値を関数に送出するように要求します。
    例外はキャッチされ、実行結果を返すイテレータを生成します。
    関数呼び出し時に要素がtupleまたはdictの場合、要素はデフォルトでアンパックされます。
    関数に非同期関数も渡すことができます。

    クエリの非同期実行についてを参照ください。

    Args:

    * self: フィルタ対象のシーケンス
    * func: 値の送出先の関数
    * unpack_kw: 値をアンパックする

    Returns: 実行結果を含むタプル

    Usage:
    ```
    >>> def do_something(id, val):
    >>>   if val:
    >>>     return 1
    >>>   else:
    >>>     raise ValueError(val)
    >>>
    >>> for elm, err, result, *_ in pnq.query([{"id": 1, "val": True}, {"id": 1, "val": False}]).request(do_something):
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


async def request_async(
    self, func, unpack: bool = True, timeout: float = None, retry: int = None
):
    for elm in self:
        err = None
        result = None
        try:
            result = await func(elm)
        except Exception as err:  # noqa
            pass

        yield elm, err, result


@mark
async def request_gather(
    self,
    func,
    unpack: bool = True,
    timeout: float = None,
    retry: int = None,
    pool: int = 3,
):
    pass


###########################################
# Expander set operation
###########################################

# https://docs.aws.amazon.com/ja_jp/redshift/latest/dg/r_UNION.html


@mark
def union_all(self):
    """全ての行を集合に含む"""
    pass


extend = union_all


@mark
def union(self, *iterables):
    """全ての行を集合に含み重複は許可しない"""
    pass


@mark
def union_intersect(self):
    """共通部分のみ抽出"""
    pass


@mark
def union_minus(self):
    """1つ目の問い合わせには存在するが、2つ目の問い合わせには存在しないデータを抽出
    exceptと同じ意味
    """
    pass


# difference
# symmetric_difference


@mark
@name_as("zip")
def __zip(self, *iterables):
    raise NotImplementedError()


###########################################
# filtering
###########################################
@mark
@name_as("filter")
def __filter(self, predicate):
    """述語に基づいてシーケンスの要素をフィルタ処理します。

    Args:

    * self: フィルタ対象のシーケンス
    * predicate: 条件を満たすか検証する関数

    Returns: 条件を満たす要素を返すクエリ

    Usage:
    ```
    >>> pnq.query([1, 2]).filter(lambda x: x == 1).to(list)
    [1]
    >>> pnq.query({1: True, 2: False, 3: True}).filter(lambda x: x[1] == True).to(list)
    [(1, True), (3, True)]
    ```
    """
    return filter(predicate, self)


@mark
# セレクタ引数はなし。可変長引数にセレクタを放り込んでしまったりややこしい。
def filter_in(self, *values):
    pass


@mark
def must(self, predicate):
    """述語に基づいてシーケンスの要素を検証します。
    検証に失敗した場合、即時に例外が発生します。

    Args:

    * self: フィルタ対象のシーケンス
    * predicate: 条件を満たすか検証する関数

    Returns: 全要素を返すクエリ（例外が発生しない限り）

    Usage:
    ```
    >>> pnq.query([1, 2]).must(lambda x: x == 1).to(list)
    raise ValueError("2")
    >>> pnq.query({1: True, 2: False, 3: True}).must(lambda x: x[1] == True).to(list)
    raise ValueError("(2, False)")
    ```
    """
    for elm in self:
        if not predicate(elm):
            raise ValueError(elm)
        yield elm


@mark
def filter_type(self, *types):
    """指定した型に一致するシーケンスの要素をフィルタ処理します。
    型は複数指定することができ、`isinstance`の挙動に準じます。

    Args:

    * self: フィルタ対象のシーケンス
    * types: フィルタする型

    Returns: 指定した型の要素を返すクエリ

    Usage:
    ```
    >>> pnq.query([1, False, "a"]).filter_type(int).to(list)
    [1, False]
    >>> pnq.query([1, False, "a"]).filter_type(str, bool).to(list)
    [False, "a"]
    ```
    """
    for elm in self:
        if isinstance(elm, *types):
            yield elm


@mark
def must_type(self, *types):
    """シーケンスの要素が指定した型のいずれかであるか検証します。
    検証に失敗した場合、即時に例外が発生します。
    型は複数指定することができ、`isinstance`の挙動に準じます。

    Args:

    * self: フィルタ対象のシーケンス
    * types: フィルタする型

    Returns: 全要素を返すクエリ（例外が発生しない限り）

    Usage:
    ```
    >>> pnq.query([1, 2]).must_type(str, int).to(list)
    raise ValueError("1 is not str")
    ```
    """
    for elm in self:
        if not isinstance(elm, *types):
            raise TypeError(f"{elm} is not {types}")
        yield elm


@mark
def unique(self, selector):
    """シーケンスの要素から重複する要素を除去する。
    セレクタによって選択された値に対して重複が検証され、その値を返す。

    Args:

    * self: フィルタ対象のシーケンス
    * selector: 重複を検証する値（複数の値を検証する場合はタプル）

    Returns: 重複を含まない要素を返すクエリ

    Usage:
    ```
    >>> pnq.query([1, 2, 1]).filter_unique().to(list)
    [1, 2]
    >>> pnq.query([(0 , 0 , 0), (0 , 1 , 1), (0 , 0 , 2)]).unique(lambda x: (x[0], x[1])).to(list)
    [(0, 0), (0, 1)]
    ```
    """
    duplidate = set()
    for elm in self:
        value = selector(elm)
        if value in duplidate:
            pass
        else:
            duplidate.add(value)
            yield elm


distinct = unique


@mark
def must_unique(self, selector=lambda x: x, immediate: bool = True):
    """シーケンスの要素から値を選択し、選択した値が重複していないか検証します。

    Args:

    * self: フィルタ対象のシーケンス
    * selector: 検証する値を選択する関数
    * immediate: 即時に例外を発生させる

    Returns: 全要素を返すクエリ（例外が発生しない限り）

    Usage:
    ```
    >>> pnq.query([1, 2, 1]).must_unique().to(list)
    raise DuplicateError("1")
    ```

    """
    duplidate = set()
    for elm in self:
        value = selector(elm)
        if value in duplidate:
            raise DuplicateElementError(elm)
        else:
            duplidate.add(value)
            yield elm


@mark
def get_many(self, *keys):
    """"""
    pass


@mark
def must_get_many(self, *keys):
    """"""
    pass


###########################################
# partitioning
###########################################


@mark
def take(self, count: int):
    """シーケンスの先頭から、指定した要素数を返します。

    Args:

    * self: バイパス対象のシーケンス
    * count: 取得する要素数

    Returns: 取得された要素を返すクエリ

    Usage:
    ```
    >>> pnq.query([1, 2, 3]).take(2).to(list)
    [1, 2]
    ```
    """
    pass


@mark
def take_while(self, predicate):
    """シーケンスの先頭から、条件の検証に失敗するまでの要素を返します。

    Args:

    * self: バイパス対象のシーケンス
    * predicate: 条件を検証する関数

    Returns: 取得された要素を返すクエリ

    Usage:
    ```
    >>> pnq.query([1, 2, 3]).enumerate().take_while(lambda v: v[0] < 2).select(1).to(list)
    [1, 2]
    ```
    """
    pass


@mark
def skip(self, count: int):
    """シーケンス内の指定された数の要素をバイパスし、残りの要素を返します。

    Args:

    * self: バイパス対象のシーケンス
    * count: バイパスする要素数

    Returns: 取得された要素を返すクエリ

    Usage:
    ```
    >>> pnq.query([1, 2, 3]).skip(1).to(list)
    [2, 3]
    ```
    """
    pass


@mark
def skip_while(self, predicate):
    """シーケンスの先頭から、条件の検証に失敗するまでの要素をバイパスし、残りの要素を返します。

    Args:

    * self: バイパス対象のシーケンス
    * predicate: 条件を検証する関数

    Returns: 取得された要素を返すクエリ

    Usage:
    ```
    >>> pnq.query([1, 2, 3]).enumerate().skip_while(lambda v: v[0] < 1).select(1).to(list)
    [2, 3]
    ```
    """
    pass


@mark
def take_range(self, start: int = 0, stop: int = ...):
    """シーケンスから指定した範囲の要素を返します。

    Args:

    * self: バイパス対象のシーケンス
    * start: 開始範囲（0始まり）
    * stop: 終了範囲（デフォルトは無限）

    Returns: 取得された要素を返すクエリ

    Usage:
    ```
    >>> pnq.query([0, 1, 2, 3, 4, 5]).take_range(0, 3).to(list)
    [0, 1, 2]
    >>> pnq.query([0, 1, 2, 3, 4, 5]).take_range(3).to(list)
    [3, 4, 5]
    ```
    """


@mark
def take_page(self, page: int, size: int):
    """シーケンスから指定した範囲の要素を返します。
    範囲はページサイズと取得対象のページから求められます。

    Args:

    * self: バイパス対象のシーケンス
    * page: 取得対象のページ（1始まり）
    * size: １ページあたりの要素数

    Returns: 取得された要素を返すクエリ

    Usage:
    ```
    >>> pnq.query([0, 1, 2, 3, 4, 5]).take_page(page=1, size=2).to(list)
    [0, 1]
    >>> pnq.query([0, 1, 2, 3, 4, 5]).take_page(page=2, size=3).to(list)
    [3, 4, 5]
    ```
    """


###########################################
# order
###########################################


@mark
def order_by(self, selector, desc: bool = False):
    """シーケンスの要素を昇順でソートします。

    Args:

    * self: ソート対象のシーケンス
    * selector: 要素からキーを抽出する関数。複数のキーを評価する場合は、タプルを返してください。
    * desc: 降順で並べる場合はTrue

    Usage:
    ```
    >>> pnq.query([1, 2, 3]).order_by().to(list)
    [1, 2, 3]
    >>> pnq.query([1, 2, 3]).order_by(lambda x: -x).to(list)
    [3, 2, 1]
    >>> pnq.query([1, 2, 3]).order_by(desc=True).to(list)
    [3, 2, 1]
    >>> pnq.query([(1, 2)), (2, 2), (2, 1)]).order_by(lambda x: (x[0], x[1])).to(list)
    [(1, 2)), (2, 1), (2, 2)]
    ```
    """
    yield from sorted(self, key=selector, reverse=desc)


@mark
def order_by_fields(self, *fields, desc: bool = False, attr: bool = False):
    selector = itemgetter(*items)
    selector = attrgetter(*attrs)
    return order_by(selector, desc)


@mark
def order_by_reverse(self):
    """シーケンスの要素を逆順にします。

    Args:

    * self: ソート対象のシーケンス

    Usage:
    ```
    >>> pnq.query([1, 2, 3]).order_by_reverse().to(list)
    [3, 2, 1]
    ```
    """
    yield from reversed(list(__iter(self)))


@mark
def order_by_shuffle(self):
    """シーケンスの要素をランダムにソートします。

    Args:

    * self: ソート対象のシーケンス

    Usage:
    ```
    >>> pnq.query([1, 2, 3]).order_by_shuffle().to(list)
    [1, 3, 2]
    >>> pnq.query([1, 2, 3]).order_by_shuffle().to(list)
    [3, 1, 2]
    ```
    """
    pass


###########################################
# statistics
###########################################
@mark
@name_as("len")
def __len(self):
    """シーケンスの要素数を返します。

    Usage:
    ```
    >>> pnq.query([1, 2, 3]).len()
    3
    ```
    """
    if hasattr(self, "__len__"):
        return len(self)

    it = __iter(self)
    for i in enumerate(it, 1):
        ...

    return i


@mark
def exists(self):
    """シーケンス内の要素の有無を確認します。

    Usage:
    ```
    >>> pnq.query([]).exists()
    False
    >>> pnq.query([1]).exists()
    True
    ```
    """
    it = __iter(self)
    for elm in it:
        return True

    return False


@mark
@name_as("all")
def __all(self, selector=lambda x: x):
    """シーケンスの全ての要素がTrueと判定できるか評価します。要素がない場合はTrueを返します。

    * selector: 要素から検証する値を抽出する関数

    Usage:
    ```
    >>> pnq.query([]).all()
    True
    >>> pnq.query([0]).all()
    False
    >>> pnq.query([1]).all()
    True
    >>> pnq.query([1, 0]).all()
    False
    >>> pnq.query([1, 2]).all()
    True
    ```
    """
    return all(__iter(self, selector))


@mark
@name_as("any")
def __any(self, selector=lambda x: x):
    """シーケンスのいずれかの要素がTrueと判定できるか評価します。要素がない場合はFalseを返します。

    * selector: 要素から検証する値を抽出する関数

    Usage:
    ```
    >>> pnq.query([]).any()
    False
    >>> pnq.query([0]).any()
    False
    >>> pnq.query([1]).any()
    True
    >>> pnq.query([1, 0]).any()
    True
    ```
    """
    return any(__iter(self, selector))


@mark
def contains(self, value, selector=lambda x: x) -> bool:
    """既定の等値比較子を使用して、指定した要素がシーケンスに含まれているか評価します。
    辞書をソースとした場合は、キーバリューのタプルを比較します。


    * value: 検索対象の値
    * selector: 要素から検索する値を抽出する関数

    Usage:
    ```
    >>> fruits = pnq.query(["apple", "orange"])
    >>> fruits.contains("banana")
    False
    >>> fruits.contains("apple")
    True
    >>> fruits.contains("orange")
    True
    >>> pnq.query({"a": 1, "b": 2}).contains("a")
    False
    >>> pnq.query({"a": 1, "b": 2}).contains(("a", 1))
    True
    ```
    """
    for val in __iter(self, selector):
        if val == value:
            return True

    return False


@mark
@name_as("min")
def __min(self, selector=lambda x: x, default=NoReturn):
    """シーケンスの要素から最小の値を取得します。

    Args:

    * selector: 要素から計算する値を抽出する関数
    * default: 要素が存在しない場合に返す値

    Usage:
    ```
    >>> pnq.query([1, 2]).min()
    1
    >>> pnq.query([]).min()
    ValueError: min() arg is an empty sequence
    >>> pnq.query([]).min(default=0)
    0
    ```
    """
    if default is NoReturn:
        return min(__iter(self, selector))
    else:
        return min(__iter(self, selector), default=default)


@mark
@name_as("max")
def __max(self, selector=lambda x: x, default=NoReturn):
    """シーケンスの要素から最大の値を取得します。

    Args:

    * selector: 要素から計算する値を抽出する関数
    * default: 要素が存在しない場合に返す値

    Usage:
    ```
    >>> pnq.query([1, 2]).max()
    2
    >>> pnq.query([]).max()
    ValueError: max() arg is an empty sequence
    >>> pnq.query([]).max(default=0)
    0
    ```
    """
    if default is NoReturn:
        return max(__iter(self, selector))
    else:
        return max(__iter(self, selector), default=default)


@mark
@name_as("sum")
def __sum(self, selector=lambda x: x):
    """シーケンスの要素を合計します。

    * selector: 要素から計算する値を抽出する関数

    Usage:
    ```
    >>> pnq.query([]).sum()
    0
    >>> pnq.query([1, 2]).sum()
    3
    ```
    """
    return sum(__iter(self, selector))


@mark
def average(
    self, selector=lambda x: x, exp: float = 0.00001, round: TH_ROUND = "ROUND_HALF_UP"
) -> Union[float, Decimal]:
    """シーケンスの要素の平均を求めます。

    Args:

    * selector: 要素から計算する値を抽出する関数
    * exp: 丸める小数点以下の桁数
    * round: 丸め方式

    Usage:
    ```
    >>> pnq.query([]).average()
    0
    >>> pnq.query([1, 2]).average()
    1.5
    ```
    """
    # import statistics
    # return statistics.mean(pmap(self, selector))  # type: ignore

    seed = Decimal("0")
    i = 0
    val = 0

    it = __iter(self, selector)
    for i, val in enumerate(it, 1):
        # val = selector(val)
        try:
            val = Decimal(str(val))  # type: ignore
        except DecimalInvalidOperation:
            raise TypeError(f"{val!r} is not a number")
        seed += val

    if i:
        result = seed / i
    else:
        result = seed

    result = result.quantize(Decimal(str(exp)), rounding=round)

    if isinstance(val, (int, float)):
        return float(result)
    else:
        return result


@mark
def reduce(
    self,
    seed: T,
    op: Union[TH_ASSIGN_OP, Callable[[Any, Any], Any]] = "+=",
    selector=lambda_empty,
) -> T:
    """シーケンスの要素を指定した代入演算子でシードに合成し、合成結果を返す。

    Args:

    * seed: 合成対象とする初期値(左辺)
    * op: 代入演算子または２項演算関数
    * selector: 要素から結合する値を抽出する関数（右辺）


    Usage:
    ```
    >>> pnq.query([1]).reduce(10, "+=")
    11
    >>> pnq.query([[1, 2, 3], [4, 5, 6]]).reduce([], "+=")
    [1, 2, 3, 4, 5, 6]
    >>> pnq.query([{"a": 1}, {"b": 2}]).reduce({}, "|=") # python3.9~
    {"a": 1, "b": 2}
    >>> pnq.query([1, 2, 3, 4, 5]).reduce(0, "+=", lambda x: x * 10)
    150
    >>> pnq.query([1, 2, 3, 4, 5]).reduce(0, lambda l, r: l + r, lambda x: x * 10)
    150
    ```
    """
    if callable(op):
        binary_op = op
    else:
        binary_op = MAP_ASSIGN_OP[op]

    for val in __iter(self, selector):
        seed = binary_op(seed, val)

    return seed


@mark
def concat(self, selector=lambda x: x, delimiter: str = ""):
    """シーケンスの要素を文字列として連結します。

    Args:

    * selector: 要素から結合する値を抽出する関数
    * delimiter: 区切り文字

    Usage:
    ```
    >>> pnq.query([]).concat()
    ""
    >>> pnq.query([1, 2]).concat()
    "12"
    >>> pnq.query(["a", "b"]).concat()
    "ab"
    >>> pnq.query(["a", None]).concat()
    "aNone"
    >>> pnq.query(["a", "b"]).concat(delimiter=",")
    "a,b"
    ```
    """
    return delimiter.join(str(x) for x in __iter(self, selector))


###########################################
# finalizer
###########################################


@mark
def to(self, finalizer):
    """クエリを即時評価し、評価結果をファイナライザによって処理します。

    Args:

    * self: バイパス対象のシーケンス
    * finalizer: イテレータを受け取るクラス・関数

    Returns: ファイナライザが返す結果

    Usage:
    ```
    >>> pnq.query([1, 2, 3]).to(list)
    [1, 2]
    >>> pnq.query({1: "a", 2: "b"}).to(dict)
    {1: "a", 2: "b"}
    ```
    """
    return finalizer(x for x in __iter(self))


@mark
def lazy(self, finalizer):
    """ファイナライザの実行するレイジーオブジェクトを返します。
    レイジーオブジェクトをコールすると同期実行され、`await`すると非同期実行されます。

    Args:

    * self: バイパス対象のシーケンス
    * finalizer: イテレータを受け取るクラス・関数

    Returns: ファイナライザが返す結果

    Usage:
    ```
    >>> lazy = pnq.query([1, 2, 3]).lazy(list)
    >>> lazy()
    [1, 2, 3]
    >>> await lazy
    [1, 2, 3]
    >>> lazy = pnq.query([1, 2, 3]).lazy(pnq.actions.first)
    >>> await lazy
    1
    ```
    """
    return finalizer(x for x in __iter(self))


@mark
async def to_async(self, cls):
    """クエリを即時評価し、評価結果をファイナライザによって処理します。
    クエリが非同期処理を要求する場合のみ使用してください。

    Args:

    * self: バイパス対象のシーケンス
    * finalizer: イテレータを受け取るクラス・関数

    Returns: ファイナライザが返す結果

    Usage:
    ```
    >>> await pnq.query([1, 2, 3]).to_async(list)
    [1, 2]
    >>> await pnq.query({1: "a", 2: "b"}).to_async(dict)
    {1: "a", 2: "b"}
    ```
    """
    return cls(x async for x in self)


@mark
def dispatch(self, func, unpack: bool = True):
    """シーケンスから流れてくる値を関数に送出します。
    デフォルトで、要素から流れてくる値をアンパックして関数に送出します。
    例外はコントロールされません。
    関数を指定しない場合、単にイテレーションを実行します。

    Args:

    * self: フィルタ対象のシーケンス
    * func: 値の送出先の関数
    * unpack_kw: 値をキーワードアンパックする

    Returns: `None`

    Usage:
    ```
    >>> pnq.query([1,2]).dispatch()
    >>> pnq.query([1,2]).dispatch(print)
    1
    2
    >>> @pnq.query([{"v1": 1, "v2": 2}]).dispatch
    >>> def print_values(v1, v2):
    >>>   print(v1, v2)
    >>> 1, 2
    >>> await pnq.query([1,2]).dispatch
    ```
    """
    for elm in self:
        yield func(elm)


@mark
async def dispatch_async(self, func, unpack: bool = True):
    """シーケンスから流れてくる値を非同期関数に送出します。
    デフォルトで、要素から流れてくる値をアンパックして関数に送出します。
    例外はコントロールされません。

    Args:

    * self: フィルタ対象のシーケンス
    * func: 値の送出先の関数
    * unpack_kw: 値をキーワードアンパックする

    Returns: `None`

    Usage:
    ```
    >>> results = []
    >>> await pnq.query([1,2]).dispatch_async(results.append)
    >>> print(results)
    [1, 2]
    ```
    """
    for elm in self:
        yield await func(elm)


@mark
def get(self, key, default=NoReturn):
    """リストや辞書などの`__getitem__`を呼び出します。セットでも使用でき、キーが存在する場合そのままキーを返します。
    デフォルトを設定した場合、キーが存在しない場合はデフォルトを返します。

    このメソッドは実体化している状態でのみ使用することができ、クエリ化されている状態では使用できません。

    Args:

    * key: キー
    * default: キーが存在しない場合に返すデフォルト値

    Usage:
    ```
    >>> data = pnq.query({"a", "b", "c"})
    >>> data.get("a")
    "a"
    >>> data.get("d")
    raise KeyError("d")
    >>> data.get("d", 10)
    10
    ```
    """
    try:
        return self[key]
    except (IndexError, KeyError):
        if isinstance(self, set) and key in self:
            return key

        if default is NoReturn:
            raise NotFoundError(key)
        else:
            return default


@mark
def one(self, default=NoReturn):
    """シーケンス内の要素が１つであることを検証し、その要素を返します。
    検証に失敗した場合は、例外が発生します。
    デフォルト値を設定した場合は、要素が存在しない場合にデフォルト値を返します。

    `one`関数は、１つの要素であるか検証するために２つ目の要素を取り出そうとします。
    ソースとなるイテラブルが値を消費する実装だと、２つの要素が失われる可能性があることに注意してください。

    Args:

    * default: 要素が存在しない場合に返す値

    Usage:
    ```
    >>> pnq.query([]).one()
    raise NoElementError("...")
    >>> pnq.query([1]).one()
    1
    >>> pnq.query([1, 2]).one()
    raise NotOneElementError("...")
    >>> pnq.query([]).one(None)
    None
    >>> pnq.query([1, 2]).one(None)
    raise NotOneElementError("...")
    ```
    """
    it = __iter(self)
    try:
        result = next(it)
    except StopIteration:
        raise NoElementError()

    try:
        next(it)
        raise NotOneElementError()
    except StopIteration:
        pass

    return result


@mark
def first(self, default=NoReturn):
    """シーケンス内の最初の要素を返します。
    要素が存在しない場合は、例外が発生します。

    セットをソースとした場合、セットは順序を保持しないため、順序性は期待できません。

    Args:

    * default: 要素が存在しない場合に返す値

    Usage:
    ```
    >>> pnq.query([]).first()
    raise NoElementError("...")
    >>> pnq.query([1]).first()
    1
    >>> pnq.query([1, 2]).first()
    1
    >>> pnq.query([]).first(None)
    None
    ```
    """
    it = __iter(self)
    try:
        return next(it)
    except StopIteration:
        raise NoElementError()


@mark
def last(self):
    """シーケンス内の最後の要素を返します。
    要素が存在しない場合は、例外が発生します。

    セットをソースとした場合、セットは順序を保持しないため、順序性は期待できません。

    Args:

    * default: 要素が存在しない場合に返す値

    Usage:
    ```
    >>> pnq.query([]).last()
    raise NoElementError("...")
    >>> pnq.query([1]).last()
    1
    >>> pnq.query([1, 2]).last()
    2
    >>> pnq.query([]).last(None)
    None
    ```
    """
    if isinstance(self, Sequence):
        try:
            return self[-1]
        except IndexError:
            raise NoElementError()

    undefined = object()
    last = undefined
    for elm in __iter(self):
        last = elm

    if last is undefined:
        raise NoElementError()
    else:
        return last


@mark
def get_or(self, key, default):
    try:
        return actions.get(self, key)
    except NotFoundError:
        return default


@mark
def one_or(self, default):
    try:
        return actions.one(self)
    except NoElementError:
        return default


@mark
def first_or(self, default):
    try:
        return actions.first(self)
    except NoElementError:
        return default


@mark
def last_or(self, default):
    try:
        return actions.last(self)
    except NoElementError:
        return default


@mark
def get_or_raise(self, key, exc: Exception):
    """基本的な動作は`get`を参照ください。
    KeyErrorが発生した時、任意の例外を発生させます。

    Args:

    * exc: KeyError時に発生させる例外

    Usage:
    ```
    >>> pnq.query([]).get_or_raise(0, Exception(f"Not Exist Key: 0"))
    raise Exception("Not Exist Key: 0")
    ```
    """
    pass


@mark
def one_or_raise(self, exc: Exception):
    """基本的な動作は`one`を参照ください。
    NoElementErrorが発生した時、任意の例外を発生させます。
    NotOneElementErrorはキャッチしません。

    Args:

    * exc: NoElementError時に発生させる例外

    Usage:
    ```
    >>> pnq.query([]).one_or_raise(0, Exception("No exists."))
    raise Exception("No exists.")
    ```
    """
    pass


@mark
def first_or_raise(self, exc: Exception):
    """基本的な動作は`first`を参照ください。
    NoElementErrorが発生した時、任意の例外を発生させます。

    Args:

    * exc: NoElementError時に発生させる例外

    Usage:
    ```
    >>> pnq.query([]).first_or_raise(0, Exception("No exists."))
    raise Exception("No exists.")
    ```
    """
    pass


@mark
def last_or_raise(self, exc: Exception):
    """基本的な動作は`last`を参照ください。
    NoElementErrorが発生した時、任意の例外を発生させます。

    Args:

    * exc: NoElementError時に発生させる例外

    Usage:
    ```
    >>> pnq.query([]).last_or_raise(0, Exception("No exists."))
    raise Exception("No exists.")
    ```
    """
    pass
