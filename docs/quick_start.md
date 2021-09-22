## インストール

まず、`pnq`をインストールします。

```shell
$ pip install pnq
```

## 動作確認

次のスクリプトを実行し、動作確認します。

```python
import pnq

pnq.query([1]).map(lambda x: x * 2).to(list)
# >> [2]

pnq.query({"a": 1, "b": 2}).filter(lambda x: x[0] == "a").to(list)
# >> [("a", 1)]
```

## クエリオブジェクト

`pnq.query`は、イテラブルなオブジェクトを受け取ると、さまざまなクエリ操作が可能なクエリオブジェクトを返します。
クエリオブジェクトを扱う上で、まず簡単に特性を理解しましょう。

### 1. クエリオブジェクトはイテラブルなプロトコルのみ公開します

クエリオブジェクトはリストまたは辞書のように扱うことはできず、`__iter__`もしくは`__aiter__`とコレクション操作に関するクエリメソッドのみ提供するイテラブルなオブジェクトであることを主張します。
例えば、添字アクセスは不可能になります。

``` python
data = pnq.query({"a": 1, "b": 2})
print(data["a"])
# =>  'Query' object is not subscriptable
```

辞書やリストをクエリ化した直後であれば、代わりに`get`を使用することができます。

``` python
data = pnq.query({"a": 1, "b": 2})
print(data.get("a"))
# => 1
```

### 2. 辞書はキーバリューペア（タプル）を返します

Pythonでは辞書をイテレートするとキーのみを返しますが、
`pnq.query`で辞書をラップするとキーバリューペアをイテレートするようになります。

``` python
list({"a": 1, "b": 2})
# =>  ["a", "b"]

list(pnq.query({"a": 1, "b": 2}))
# =>  [("a", 1), ("b", 2)]
```

キーバリューペアを要素とするイテラブルは`dict`関数で辞書化できるため、いつでも辞書に戻すことができます。

``` python
dict([("a", 1), ("b", 2)])
# => {"a": 1, "b": 2}
```

キーバリューペアは添字でキーと値にアクセスできます。

``` python
dict(pnq.query({"a": 1, "b": 2}).map(lambda x: (x[0], x[1] * 2)))
# =>  {"a": 2, "b": 4}
```


### 3. 非同期イテレータのサポート

クエリオブジェクトは`__aiter__`をサポートしているので、`async_for`でも同じようにクエリメソッドを使用できます。
ただし、一部メソッドは非同期未対応です。

``` python
import asyncio

async def main():
    async def async_iterate():
        yield 1
        yield 2
        yield 3

    async for x in pnq.query(async_iterate()).map(lambda x: x * 2):
        print(x)

import asyncio.run(main())
# => 2 4 6
```

### 4. 遅延評価・キャッシュ

クエリオブジェクトは、評価が必要となるまでイテレーションを保留します。
次の例を見てみます。

``` python
def add_one(x):
    print(x)
    return x + 1


q = pnq.query([1]).map(add_one)

result = q.to(list)
# => 1
```

`add_one`に定義された`print`の結果は`to(list)`のタイミングで表示され、
`map(add_one)`を定義した時点では実行されないことが分かります。

次の例も見てみます。

``` python
def add_one(x):
    print(x)
    return x + 1

q = pnq.query([1]).map(add_one)

for x in q:
    ...
# => 1

for x in q:
    ...
# => 1
```

2回の`for`のタイミングで`print`が実行されています。

ここで、2回`print`が実行されるべきか検討が必要です。
2回実行されるということは多くの処理が走っていることになります。

何度も同じ結果を使い回す意図があるなら、`to(list)`等で評価を確定させて結果をキャッシュするべきです。

``` python
def add_one(x):
    print(x)
    return x + 1

result = pnq.query([1]).map(add_one).to(list)
# => 1

for x in result:
    ...
```

## はじめよう

これで自由にクエリオブジェクトを扱えるようになったはずです。

次章の参考例からお気に入りの機能を見つけましょう。
