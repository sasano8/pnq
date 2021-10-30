import sys
from collections import defaultdict
from decimal import Decimal
from decimal import InvalidOperation as DecimalInvalidOperation
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    NoReturn,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    overload,
)

try:
    from typing import Literal
except:
    from typing_extensions import Literal

from pnq._itertools.op import MAP_ASSIGN_OP, TH_ASSIGN_OP, TH_ROUND
from pnq.exceptions import (
    DuplicateElementError,
    MustError,
    MustTypeError,
    NoElementError,
    NotFoundError,
    NotOneElementError,
)

if TYPE_CHECKING:
    from _typeshed import SupportsKeysAndGetItem


T = TypeVar("T")
K = TypeVar("K")
V = TypeVar("V")


class EmptyLambda:
    def __call__(self, x):
        return x

    def __str__(self):
        return "lambda x: x"


lambda_empty = EmptyLambda()
# lambda_empty = lambda x: x
# lambda_empty.__str__ = lambda: "lambda x: x"


def name_as(name):
    def wrapper(func):
        func.__name__ = name
        return func

    return wrapper


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


def __map_nullable(self, selector):
    if selector is None:
        return self
    else:
        return map(selector, self)


@mark
def __next(self):
    raise NotImplementedError()


@mark
def value(*args, **kwargs):
    """１つの要素を返すイテレータを生成します。
    位置引数を与えた場合は与えた値かタプルになり、キーワード引数を与えた場合は辞書になります。

    Usage:
    ```
    >>> pnq.value(1).to(list)
    [1]
    >>> pnq.value("a", "b").to(list)
    [("a", "b")]
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
def map(self, selector, unpack=""):
    """シーケンスの各要素を新しいフォームに射影します。
    str関数を渡した場合、`None`は`""`を返します（Pythonの標準動作は`"None"`を返します）。

    Args:

    * self: 変換対象のシーケンス
    * selector: 各要素に対する変換関数
    * unpack: 引数をどのように展開するか指定する
        - `""`: 展開せずにそのまま値を渡す（デフォルト）
        - `"\*"`: 位置引数として展開する
        - `"\*\*"`: キーワード引数として展開する
        - `"\*\*\*"`: 位置引数とキーワード引数を展開する（pnq.Argumentsインスタンスに対して使用できます）

    Usage:
    ```
    >>> pnq.query([1]).map(lambda x: x * 2).to(list)
    [2]
    >>> pnq.query([None]).map(str).to(list)
    [""]
    >>> pnq.query([(1, 2)]).map(lambda arg1, arg2: arg1, "*").to(list)
    [1]
    >>> pnq.query([{"arg1": 1, "arg2": 2}]).map(lambda arg1, arg2: arg1, "\*\*").to(list)
    [1]
    >>> pnq.query([pnq.Arguments(1, 2, name="test", age=20)]).map(lambda arg1, arg2, name, age: name, "\*\*\*").to(list)
    ["test"]
    ```
    """


def __map_nullable(self, selector):
    if selector is None:
        return self
    else:
        return map(selector, self)


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
def select_single_node(
    self,
    node_selector=lambda x: x.node,
    return_selector=lambda nest, parent, child: (nest, parent, child),
):
    pass


@mark
def gather(self):
    """シーケンスの要素から結果を取り出し、新しいフォームに射影します。
    要素は、`concurrent.futures.Future`、または、`awaitable`である必要があります。

    `pnq.query`は、それらのインターフェースを実装しているため、本メソッドで実体化可能です。

    Usage:
    ```
    from concurretn.futures import Future
    import asyncio

    def main1():
        future = Future()
        future.set_result(0)
        return pnq.query([future]).gather().result()

    async def main2():
        async def heavy_task():
            return 1

        async def aiter():
            yield 3
            yield 4

        future = Future()
        future.set_result(0)

        tasks = pnq.query([
            future,
            heavy_task(),
            asyncio.create_task(heavy_task()),
            pnq.([1, 2]).map(lambda x: x * 2),
            pnq.(aiter()).map(lambda x: x * 2)
        ])
        return await tasks.gather()

    main()  # => [0]
    asyncio.run(main2())  # => [[0], [1], [1], [2, 4], [6, 8]]
    ```
    """


@mark
def select(self, field, *fields, attr: bool = False):
    """シーケンスの各要素からアイテムを選択し新しいフォームに射影します。
    複数のアイテムを選択した場合は、タプルとして射影します。

    Args:

    * self: 変換対象のシーケンス
    * field: 各要素から選択するアイテム
    * fields: 各要素から選択する追加のアイテム
    * attr: 要素の属性から取得する場合はTrue

    Usage:
    ```
    >>> pnq.query([(1, 2)]).select(0).to(list)
    [1]
    >>> pnq.query([{"id": 1, "name": "a"}]).select("id", "name").to(list)
    [(1, "a")]
    >>> pnq.query([user]).select("id", "name", attr=True).to(list)
    [(1, "a")]
    ```
    """


@mark
def select_as_tuple(self, *fields, attr: bool = False):
    """シーケンスの各要素からアイテムまたは属性を選択し辞書として新しいフォームに射影します。
    selectと似ていますが、選択した値が１つでも必ずタプルを返します。

    Args:

    * self: 変換対象のシーケンス
    * fields: 選択するアイテムまたは属性
    * attr: 属性から取得する場合はTrueとする

    Usage:
    ```
    >>> pnq.query([(1, 2)]).select_as_tuple(0).to(list)
    [(1,)]
    >>> pnq.query([user]).select_as_tuple("id", "name", attr=True).to(list)
    [("1", "a")]
    ```
    """


@mark
def select_as_dict(self, *fields, attr: bool = False):
    """シーケンスの各要素からアイテムまたは属性を選択し辞書として新しいフォームに射影します。

    Args:

    * self: 変換対象のシーケンス
    * fields: 選択するアイテムまたは属性
    * attr: 属性から取得する場合はTrueとする

    Usage:
    ```
    >>> pnq.query([(1, 2)]).select_as_dict(0).to(list)
    [{0: 1}]
    >>> pnq.query([user]).select_as_dict("id", "name", attr=True).to(list)
    [{"id": 1, "name": "b"}]
    ```
    """


@mark
def reflect(self, mapping, *, default=NoReturn, attr: bool = False):
    """シーケンスの各要素のフィールドを与えたマッピングに基づき辞書として新しいフォームに射影します。

    Args:

    * self: 変換対象のシーケンス
    * mapping: 元の要素のフィールドと射影先のフィールドの対応表
    * default: フィールドを取得できない場合のデフォルト値
    * attr: 属性から取得する場合はTrueとする

    Usage:
    ```
    >>> person = {"id":1, "name": "山田", "kana": "やまだ", "note": "hoge"}
    >>> pnq.query([person]).reflect({
    >>>   "id": "id",
    >>>   "name": {"name", "searchable"},
    >>>   "kana": {"kana", "searchable"},
    >>>   "note": "searchable"
    >>> }).to(list)
    >>> [{"id": 1, "name": "山田", "kana": "やまだ", "searchable": ["山田", "やまだ", "hoge"]}]
    ```
    """


@mark
def flat(self, selector=None):
    """シーケンスの各要素をイテラブルに射影し、その結果を１つのシーケンスに平坦化します。

    Args:

    * self: 変換対象のシーケンス
    * selector: 各要素から平坦化する要素を選択する関数

    Usage:
    ```
    >>> pnq.query(["abc", "def"]).flat().to(list)
    >>> ["a", "b", "c", "d", "e", "f"]
    >>> countries = [{"country": "japan", "state": ["tokyo", "osaka"]}, {"country": "america", "state": ["new york", "florida"]}]
    >>> pnq.query(countries).flat(lambda x: x["state"]).to(list)
    >>> ["tokyo", "osaka", "new york", "florida"]
    ```
    """


select_many = flat
flat_map = flat


@mark
def traverse(self, selector):
    """シーケンスの各要素から再帰的に複数ノードを選択し、選択されたノードを１つのシーケンスに平坦化します。
    各ルート要素から浅い順に列挙されます。

    Args:

    * self: 変換対象のシーケンス
    * selector: 各要素から平坦化する要素を再帰的に選択する関数（戻り値はリスト等に含めて返す必要があります）

    Usage:
    ```
    >>> pnq.query(
    >>>     {"name": "a", "nodes": [{"name": "b", nodes: [{"name": c, "nodes": []}, {"name": "d", "nodes": []}}}]}]}
    >>> ).traverse(lambdax x: x["nodes"]).select("name").to(list)
    >>> ["a", "b", "c", "d"]
    ```
    """


def pivot_unstack(self, default=None):
    """行方向に並んでいるデータを列方向に入れ替える

    Args:

    * self: 変換対象のシーケンス
    * default: フィールドが存在しない場合のデフォルト値

    Usage:
    ```
    data = [
        {"name": "test1", "age": 20},
        {"name": "test2", "age": 25},
        {"name": "test3", "age": 30, "sex": "male"},
    ]
    {'name': ['test1', 'test2', 'test3'], 'age': [20, 25, 30], 'sex': [None, None, 'male']}
    ```
    """
    ...


def pivot_stack(self):
    """列方向に並んでいるデータを行方向に入れ替える

    Args:

    * self: 変換対象のシーケンス

    Usage:
    ```
    {'name': ['test1', 'test2', 'test3'], 'age': [20, 25, 30], 'sex': [None, None, 'male']}
    data = [
        {"name": "test1", "age": 20, "sex": None},
        {"name": "test2", "age": 25, "sex": None},
        {"name": "test3", "age": 30, "sex": "male"},
    ]
    ```
    """


@mark
def cast(self, type):
    """シーケンスの型注釈を変更します。この関数はエディタの型解釈を助けるためだけに存在し、何も処理を行いません。

    実際に型を変更する場合は、`map`を使用してください。

    Args:

    * self: 変換対象のシーケンス
    * type: 新しい型注釈

    Usage:
    ```
    >>> pnq.query([1]).cast(float)
    ```
    """


@mark
def enumerate(self, start: int = 0, step: int = 1):
    """シーケンスの各要素とインデックスを新しいフォームに射影します。

    Args:

    * self: 変換対象のシーケンス
    * start: 開始インデックス
    * step: 増分

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


@mark
def group_by(self, selector=lambda x: x):
    """シーケンスの各要素からセレクタ関数でキーとバリューを取得し、キーでグループ化されたシーケンスを生成します。
    セレクタ関数を指定しない場合、各要素がすでにキーバリューのタプルであることを期待し、キーでグループ化します。

    Args:

    * self: 変換対象のシーケンス
    * selector: キーとバリューを選択する関数

    Usage:
    ```
    >>> data = [
    >>>   {"name": "banana", "color": "yellow", "count": 3},
    >>>   {"name": "apple", "color": "red", "count": 2},
    >>>   {"name": "strawberry", "color": "red", "count": 5},
    >>> ]
    >>> pnq.query(data).group_by(lambda x: x["color"], x["name"]).to(list)
    [("yellow", ["banana"]), ("red", ["apple", "strawberry"])]
    >>> pnq.query(data).select("color", "count").group_by().to(dict)
    {"yellow": [3], "red": [2, 5]}
    ```
    """


@mark
def chunk(self, size: int):
    """
    シーケンスを指定したサイズ毎に分割する。
    """


@mark
def tee(self, size: int):
    """
    ひとつのイテレータから独立したN個のイテレータを返す
    """


@mark
def join(self, right, on, select):
    pass


@mark
def group_join(self, right, on, select):
    pass


@mark
def parallel(self, func, executor=None, *, unpack="", chunksize=1):
    """シーケンスの要素を任意の関数で並列処理し、新しいフォームに射影します。
    例外が発生した場合、後続の要素はスケジューリングされません。
    その場合、実行中の処理とキューに積まれた処理の後処理は、エクゼキュータの挙動に依存します。
    詳しくは並列処理を参照ください。

    Args:

    * func: 実行する任意の関数
    * executor: 処理を実行するエクゼキュータ
    * unpack: 引数をどのように展開するか指定する
    * chunksize: cpuバウンドなエクゼキュータで有効です。指定したサイズの要素を一括処理します。

    """


@mark
def request(
    self,
    func,
    executor=None,
    *,
    unpack="",
    chunksize=1,
    # retry: int = None,
    # timeout: float = None
):
    """シーケンスの要素を任意の関数で並列処理し、新しいフォームに射影します。
    戻り値または例外は、`asyncio.Future`互換の`Response`オブジェクトに格納されます。
    詳しくは並列処理を参照ください。

    Args:

    * func: 実行する任意の関数
    * executor: 処理を実行するエクゼキュータ
    * unpack: 引数をどのように展開するか指定する
    * chunksize: cpuバウンドなエクゼキュータで有効です。指定したサイズの要素を一括処理します。

    Usage:
    ```
    >>> def do_something(id, val):
    >>>   if val:
    >>>     return 1
    >>>   else:
    >>>     raise ValueError(val)
    >>>
    >>> for res in pnq.query([{"id": 1, "val": True}, {"id": 2, "val": False}]).request(do_something):
    >>>   if res.err:
    >>>     print(f"ERROR: {res.to(dict)}")
    >>>   else:
    >>>     print(f"SUCCESS: {res.to(dict)}")
    ```
    """


@mark
def dispatch(source, func, executor=None, *, unpack="", chunksize=1, on_complete=None):
    """シーケンスの要素を任意の関数で並列処理します。
    スケジューリングされた処理はバックグラウンドで実行され、処理結果は`on_complete`に指定した関数で受け取れます。
    アプリケーションが終了する時、バックグランドの残処理がどのように扱われるかはエクゼキュータの挙動に依存します。
    `on_complete`が確実に呼び出される保証はありません。
    詳しくは並列処理を参照ください。

    Args:

    * func: 実行する任意の関数
    * executor: 処理を実行するエクゼキュータ
    * unpack: 引数をどのように展開するか指定する
    * chunksize: cpuバウンドなエクゼキュータで有効です。指定したサイズの要素を一括処理します。
    * on_complete(x): 処理結果を受け取る関数を指定します

    """


def debug(self, breakpoint=lambda x: x, printer=print):
    for v in self:
        printer(v)
        breakpoint(v)
        yield v


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

# < <= > >= inclusion


def zip(self, *iterables):
    raise NotImplementedError()


def compress(self, *iterables):
    """未実装"""
    import itertools

    return itertools.compress(self, *iterables)


@mark
def cartesian(self, *iterables):
    """複数のシーケンスのデカルト積を返します。

    Args:

    * self, iterables: デカルト積を求める複数のシーケンス
    """


"""
# product（デカルト積）
from itertools import product
mark=['♠', '♥', '♦', '♣']
suu = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
tramp = [m for m in product(mark,suu)]
print(tramp)
# [('♠', 'A'), ('♠', '2'), ...]
"""


###########################################
# filtering
###########################################
def filter(self, predicate):
    """述語に基づいてシーケンスの要素をフィルタ処理します。

    Args:

    * self: フィルタ対象のシーケンス
    * predicate: 条件を満たすか検証する関数

    Usage:
    ```
    >>> pnq.query([1, 2]).filter(lambda x: x == 1).to(list)
    [1]
    >>> pnq.query({1: True, 2: False, 3: True}).filter(lambda x: x[1] == True).to(list)
    [(1, True), (3, True)]
    ```
    """


@mark
def must(self, predicate, msg=""):
    """述語に基づいてシーケンスの要素を検証します。
    検証に失敗した場合、即時に例外が発生します。

    Args:

    * self: フィルタ対象のシーケンス
    * predicate: 条件を満たすか検証する関数

    Usage:
    ```
    >>> pnq.query([1, 2]).must(lambda x: x == 1).to(list)
    raise ValueError("2")
    >>> pnq.query({1: True, 2: False, 3: True}).must(lambda x: x[1] == True).to(list)
    raise ValueError("(2, False)")
    ```
    """


@mark
def filter_type(self, *types):
    """指定した型に一致するシーケンスの要素をフィルタ処理します。
    型は複数指定することができ、`isinstance`の挙動に準じます。

    Args:

    * self: フィルタ対象のシーケンス
    * types: フィルタする型

    Usage:
    ```
    >>> pnq.query([1, False, "a"]).filter_type(int).to(list)
    [1, False]
    >>> pnq.query([1, False, "a"]).filter_type(str, bool).to(list)
    [False, "a"]
    ```
    """


@mark
def must_type(self, types):
    """シーケンスの要素が指定した型のいずれかであるか検証します。
    検証に失敗した場合、即時に例外が発生します。
    型は複数指定することができ、`isinstance`の挙動に準じます。

    Args:

    * self: フィルタ対象のシーケンス
    * types: フィルタする型

    Usage:
    ```
    >>> pnq.query([1, 2]).must_type(str, int).to(list)
    raise ValueError("1 is not str")
    ```
    """


@mark
def filter_keys(self, *keys):
    """シーケンスの要素から指定したキーの要素のみフィルタ処理します。
    このメソッドは、`list` `dict` `set`などをクエリ化した直後のみ利用できます。

    * `list`、`tuple`の場合インデックスでフィルタされ、値を返します。
    * `dict`の場合キーでフィルタされ、キーと要素を返します。
    * `set`の場合は、キーでフィルタされ、キーを返します。

    Args:

    * self: フィルタ対象のシーケンス
    * keys: フィルタするキー

    Usage:
    ```
    >>> pnq.query([1, 2]).filter_keys(1).to(list)
    [2]
    >>> pnq.query({"a": 1, "b": 2}).filter_keys("b").to(list)
    [("b", 2)]
    >>> pnq.query({"a", "b"}).filter_keys("b").to(list)
    ["b"]
    ```
    """


@mark
def must_keys(self, *keys):
    """`filter_keys`を実行し、全てのキーを取得できなかった場合例外を発生させます。
    検証が完了するまで、ストリームは保留されます。
    """


@mark
def filter_unique(self, selector=None):
    """シーケンスの要素から重複する要素を除去する。
    セレクタによって選択された値に対して重複が検証され、その値を返す。

    Args:

    * self: フィルタ対象のシーケンス
    * selector: 重複を検証する値（複数の値を検証する場合はタプル）

    Usage:
    ```
    >>> pnq.query([1, 2, 1]).filter_unique().to(list)
    [1, 2]
    >>> pnq.query([(0 , 0 , 0), (0 , 1 , 1), (0 , 0 , 2)]).unique(lambda x: (x[0], x[1])).to(list)
    [(0, 0), (0, 1)]
    ```
    """


distinct = filter_unique


@mark
def must_unique(self, selector=None):
    """シーケンスの要素から値を選択し、選択した値が重複していないか検証します。

    Args:

    * self: フィルタ対象のシーケンス
    * selector: 検証する値を選択する関数
    * immediate: 即時に例外を発生させる

    Usage:
    ```
    >>> pnq.query([1, 2, 1]).must_unique().to(list)
    raise DuplicateError("1")
    ```

    """


###########################################
# partitioning
###########################################


@mark
def take(self, count_or_range: int):
    """シーケンスから指定した範囲の要素を返します。

    Args:

    * self: 取得対象のシーケンス
    * count_or_range: シーケンスの先頭から取得する要素数または取得する範囲

    Usage:
    ```
    >>> pnq.query([1, 2, 3]).take(2).to(list)
    [1, 2]
    >>> pnq.query([1, 2, 3]).take(range(1, 2)).to(list)
    [2]
    ```
    """


@mark
def take_while(self, predicate):
    """シーケンスの先頭から、条件の検証に失敗するまでの要素を返します。
    **検証に失敗した要素は破棄されるため、値を消費するイテレータがソースの場合は注意が必要です。**

    Args:

    * self: バイパス対象のシーケンス
    * predicate: 条件を検証する関数

    Usage:
    ```
    >>> pnq.query([1, 2, 3]).enumerate().take_while(lambda v: v[0] < 2).select(1).to(list)
    [1, 2]
    ```
    """


@mark
def skip(self, count_or_range: int):
    """シーケンスから指定した範囲の要素をバイパスします。

    Args:

    * self: バイパス対象のシーケンス
    * count_or_range: シーケンスの先頭からバイパスする要素数またはバイパスする範囲

    Usage:
    ```
    >>> pnq.query([1, 2, 3]).skip(1).to(list)
    [2, 3]
    >>> pnq.query([1, 2, 3]).skip(range(1, 2)).to(list)
    [1, 3]
    ```
    """


@mark
def skip_while(self, predicate):
    """シーケンスの先頭から、条件の検証に失敗するまでの要素をバイパスし、残りの要素を返します。

    Args:

    * self: バイパス対象のシーケンス
    * predicate: 条件を検証する関数

    Usage:
    ```
    >>> pnq.query([1, 2, 3]).enumerate().skip_while(lambda v: v[0] < 1).select(1).to(list)
    [2, 3]
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
def order_by(self, key_selector=None, desc: bool = False):
    """シーケンスの要素を昇順でソートします。

    Args:

    * self: ソート対象のシーケンス
    * selector: 要素からキーを抽出する関数。複数のキーを評価する場合は、タプルを返してください。
    * desc: 降順で並べる場合はTrue

    Usage:
    ```
    >>> pnq.query([3, 2, 1]]).order_by().to(list)
    [1, 2, 3]
    >>> pnq.query([1, 2, 3]).order_by(lambda x: -x).to(list)
    [3, 2, 1]
    >>> pnq.query([1, 2, 3]).order_by(desc=True).to(list)
    [3, 2, 1]
    >>> pnq.query([(1, 2)), (2, 2), (2, 1)]).order_by(lambda x: (x[0], x[1])).to(list)
    [(1, 2)), (2, 1), (2, 2)]
    ```
    """


@mark
def order_by_fields(self, *fields, desc: bool = False, attr: bool = False):
    raise NotImplementedError()


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


###########################################
# statistics
###########################################
def len(self):
    """シーケンスの要素数を返します。

    Usage:
    ```
    >>> pnq.query([1, 2, 3]).len()
    3
    ```
    """


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


def all(self, selector=lambda x: x):
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


def any(self, selector=lambda x: x):
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


def min(self, selector=lambda x: x, default=NoReturn):
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


def max(self, selector=lambda x: x, default=NoReturn):
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


def sum(self, selector=lambda x: x):
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


def reduce(
    self,
    seed: T,
    op="+=",
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


def concat(self, selector=lambda x: x, delimiter: str = ""):
    """シーケンスの要素を文字列として連結します。
    Noneは空文字として扱われます。

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
    "a"
    >>> pnq.query(["a", "b"]).concat(delimiter=",")
    "a,b"
    ```
    """


###########################################
# finalizer
###########################################


@mark
def result(self):
    """ストリームを評価し、結果をリストとして保存します。
    返されたリストは、クエリメソッドが実装されたリストの拡張クラスです。

    Args:

    * self: 評価するシーケンス

    Returns: `pnq.list`

    Usage:
    ```
    >>> saved = pnq.query([1, 2, 3]).result()
    [1, 2, 3]
    >>> saved.map(lambda x: x * 2).result()
    [2, 4, 6]
    ```
    """


@mark
def save(self):
    """`result`のエイリアスです。"""


@mark
def to(self, finalizer):
    """ストリームをファイナライザによって処理します。

    Args:

    * self: 評価するシーケンス
    * finalizer: イテレータを受け取るクラス・関数

    Returns: ファイナライザが返す結果

    Usage:
    ```
    >>> pnq.query([1, 2, 3]).to(list)
    [1, 2, 3]
    >>> pnq.query({1: "a", 2: "b"}).to(dict)
    {1: "a", 2: "b"}
    ```
    """


async def to_async(self, finalizer):
    if self.is_async:
        return await finalizer(self)
    else:
        return await finalizer(self)


@mark
def to_task(self):
    """実体化を開始し、任意のタイミングで結果を受け取れるasyncio.Taskを返します。
    このメソッドは、非同期イテレータや並列処理などを非同期で進行しながら、
    他の処理を進める場合に役に立ちます。

    taskはawaitすることで、結果を受け取ることができます。

    Usage:
    ```
    >>> task = pnq.query(aiter).to_task()
    >>> await do_something()
    >>> await task
    [1, 2, 3]
    ```
    """


@mark
def to_lifespan(self):
    """イテレータを生存期間つきのイテレータ（コンテキスト）にします。
    このメソッドは、無限イテレータの開始・停止の管理に役立ちます。

    callbackが設定されている場合は、ルートイテレータが停止されるまで生存し、
    callbackが指定されていない場合は、直ちにイテレートを停止します。

    >>> queue = pnq.Queue()
    >>> async with pnq.query(queue).to_lifespan(print, callback=queue.stop):
    >>>   await queue.put(1)
    >>>   await queue.put(2)
    """


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
    return finalizer(x for x in self)


async def to_async(self, cls):
    """非同期ストリームをファイナライザによって処理します。
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


def each(self, func=lambda x: x, unpack=""):
    """シーケンスの各要素を指定した関数で逐次的に処理します。
    `for x in iterable: ...`のショートカットとして機能します。
    関数を指定しない場合、単にイテレーションを実行します。

    非同期関数を実行するには、クエリを非同期化する必要があります。

    Args:

    * func: 要素を処理する関数
    * unpack: 引数をどのように展開するか指定する

    Returns: `None`

    Usage:
    ```
    >>> pnq.query([1,2]).each()
    >>> pnq.query([1,2]).each(print)
    1
    2
    >>> await pnq.query([1,2]).\_.each(sync_func)
    >>> await pnq.query([1,2]).\_.each(async_func)
    ```
    """


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
    except (IndexError, KeyError, NotFoundError):
        pass

    except Exception:
        if isinstance(self, set) and key in self:
            return key

    if default is NoReturn:
        raise NotFoundError(key)
    else:
        return default


@mark
def one(self):
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


@mark
def first(self):
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


@mark
def get_or(self, key, default):
    try:
        return get(self, key)
    except NotFoundError:
        return default


@mark
def one_or(self, default):
    try:
        return one(self)
    except NoElementError:
        return default


@mark
def first_or(self, default):
    try:
        return first(self)
    except NoElementError:
        return default


@mark
def last_or(self, default):
    try:
        return last(self)
    except NoElementError:
        return default


@mark
def get_or_raise(self, key, exc: Union[str, Exception]):
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


@mark
def one_or_raise(self, exc: Union[str, Exception]):
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


@mark
def first_or_raise(self, exc: Union[str, Exception]):
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


@mark
def last_or_raise(self, exc: Union[str, Exception]):
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


###########################################
# sleeping
###########################################
def sleep(self, seconds: float):
    from time import sleep as sleep_


async def sleep_async(self, seconds: float):
    from asyncio import sleep as sleep_
