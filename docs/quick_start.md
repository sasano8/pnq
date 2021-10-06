## インストール

まず、`pnq`をインストールします。

```shell
$ pip install pnq
```

## クエリを組み立てる

`pnq.query`を介して、データソースを加工するパイプラインメソッドをチェインできます。
組み立てたクエリは、`save`を実行することでリストとして実体化できます。

``` python
import pnq

pnq.query([1, 2]).map(lambda x: x * 2).filter(lambda x: x > 2).save()
# => [4]
```

`save`で返されるリストは、リストを継承した独自拡張クラス（`pnq.list`）で、引き続きパイプラインメソッドをチェインできます。

``` python
import pnq

saved = pnq.query([1]).map(lambda x: x * 2).save()
saved.map(lambda x: x * 2).save()
# => [4]
```

`pnq.list`はリストと完全な互換性がありますが、可能な限り副作用を避ける場合は、`to(list)`または単に`list`で組込みのリストにできます。

``` python
import pnq

pnq.query([1]).map(lambda x: x * 2).to(list)
# => [2]

list(pnq.query([1]).map(lambda x: x * 2))
# => [2]
```

データソースが辞書の場合は、キーバリューペアが列挙されることに注意してください。

``` python
import pnq

pnq.query({"a": 1, "b": 2, "c": 3}).filter(lambda x: x[1] > 1).save()
# => [("b", 2), ("c", 3)]
```

リストでなく辞書として実体化したい場合は、`save`の代わりに`to(dict)`または単に`dict`を使用してください。

``` python
import pnq

pnq.query({"a": 1, "b": 2, "c": 3}).filter(lambda x: x[1] > 1).to(dict)
# => {"b": 2, "c": 3}

dict(pnq.query({"a": 1, "b": 2, "c": 3}).filter(lambda x: x[1] > 1))
# => {"b": 2, "c": 3}
```

なお、`to`はイテラブルを引数とする任意の関数を渡すことができます。

## 非同期イテレータを扱う

`pnq.query`は非同期イテレータも取り扱うことができます。
ただし、非同期イテレータを実体化するには`save`の代わりに`await`を使用します。

``` python
import asyncio
import pnq

async def aiter():
    yield 1
    yield 2
    yield 3

async def main():
    return await pnq.query(aiter()).map(lambda x: x * 2)

asyncio.run(main())
# >> [2, 4, 6]
```

クエリは`for`文でも使用できます。

``` python
import asyncio
import pnq

async def aiter():
    yield 4
    yield 5
    yield 6

async def main():
    for x in pnq.query([1, 2, 3]).map(lambda x: x * 2):
        print(x)
    # => 2, 4, 6

    async for x in pnq.query(aiter()).map(lambda x: x * 2):
        print(x)
    # => 8, 10, 12

asyncio.run(main())
```

## クエリを実行する

`pnq.query`は可能な限り評価を保留（遅延評価）します。
クエリは、評価を要求されたとき実際に実行されます。

すでにいくつか評価方法（`for`文、`save`、`to`）を紹介していますが、ほかにもいくつか評価メソッドを紹介します。

``` python
import pnq

# for x in ...: func(x)のショートカットとして使用できます
pnq.query([1, 2, 3]).map(lambda x: x * 2).each(print)
# => 2, 4, 6

# 要素の合計を求めます
pnq.query([1, 2, 3]).map(lambda x: x * 2).sum()
# => 12
```

非同期イテレータをデータソースとする場合は、`_`で明示的に非同期イテレータを評価すると伝え、`await`する必要があります。

``` python
import asyncio
import pnq

async def aiter():
    yield 1
    yield 2
    yield 3

async def main():
    await pnq.query(aiter()).map(lambda x: x * 2)._.each(print)
    # => 2, 4, 6

    await pnq.query(aiter()).map(lambda x: x * 2)._.sum()
    # => 12

asyncio.run(main())
```

## バッチ処理に活用する

`request`メソッドは、簡易的なバッチ処理に活用できます。
`request`メソッドはシーケンスの要素を任意の関数に送出し、実行結果（`pnq.Response`）を返します。

処理中に例外が発生した場合、例外情報が`err` `msg` `stack_trace`属性にエラー情報が格納されます。

``` python
import datetime
import logging
import pnq

log_name = "log_" + datetime.datetime.utcnow().isoformat() + ".jsonl.log"
log = logging.FileHandler(filename=log_name)

logger = logging.getLogger()
logger.addHandler(log)


params = pnq.query([{"val": 0}, {"val": 1}])

# パラメータを関数に渡します
# パラメータはキーワード引数としてアンパックされるため、パラメータは辞書互換オブジェクトである必要があります
@params.request
def do_something(val):
    if not (val > 0):
        raise ValueError(f"val must be 1 or greater. But got {val}")
    else:
        return "success"


# 処理が失敗した場合、実行情報をjsonl（１行１Json）形式で出力します
@do_something.each
def dump_if_error(x: pnq.Response):
    # エラーだった場合、ログに出力します
    if x.err:
        # レスポンスををjsonにシリアライズします
        # シリアライザはデフォルトで`json.dumps`(ensure_ascii=False)が使用されます
        logger.error(x.to_json())


# 全ての処理が成功した場合は`0`、いずれかが失敗した場合は`1`を返します
exit(pnq.from_jsonl(log_name).exists())
```

エラーログは、次のような出力になります。

``` bash
cat `ls *.jsonl.log`
# {"func": "do_something", "kwargs": {"val": 0}, "err": "ValueError", "msg": "val must be 1 or greater: 0", "result": None, ...}
```

## もっと知りたい

これであなたはクエリを自由に扱えるようになったはずです。

次章の参考例からお気に入りの機能を見つけましょう。
