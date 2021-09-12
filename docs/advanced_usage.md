## 辞書の正規化

リストと辞書の`__iter__`はそれぞれ値とキーを返すという挙動の違いがあります。

``` python
for val in [5, 6, 7]:
  print(val)
# => 5
# => 6
# => 7
```

``` python
for val in {"a": 1, "b": 2, "c": 3}:
  print(val)
# => "a"
# => "b"
# => "c"
```

`pnq`は、リストと辞書の挙動を揃えるため次のような振る舞いをします。


`pnq.query`でラップされた辞書は辞書のように振る舞います。

``` python
for val in pnq.query({"a": 5, "b": 6, "c": 7}):
    print(val)

# => "a"
# => "b"
# => "c"
```

クエリメソッドを介するとキーバリューペアを列挙するように動作します。


``` python
for val in pnq.query({"a": 5, "b": 6, "c": 7}).filter(lambda x: True):
    print(val)

# => ("a", 5)
# => ("b", 6)
# => ("c", 7)
```

例えば、次のコードは異なる結果を返します。

``` python
data = {"a": 5, "b": 6, "c": 7}

result_1 =  pnq.query(data).to(list)
# => [("a", 5), ("b", 6), ("c", 7)]

result_1 =  list(pnq.query(data))
# => ["a", "b", "c"]
```


クエリメソッドでチェーンされたオブジェクトは、イテレーションを要求されると内部的に`__piter__`を呼び出します。


``` python
for val in pnq.query({"a": 5, "b": 6, "c": 7}).__piter__():
    print(val)

# => ("a", 5)
# => ("b", 6)
# => ("c", 7)
```

!!! warning
    - `pnq.query`で受け取った直後の辞書と、一度でもクエリメソッドをチェーンした場合に列挙される要素は異なることを意識する必要があります


## ステートレスイテレータ

`pnq`はステートレスなイテレータとして動作するように設計されています。

Pythonの組込み関数にはいくつかイテレータを生成する機能がありますが、次の例を見てみましょう。

``` python
it = filter(lambda x: True, [1, 2, 3])

result_1 = list(it)
# => [1, 2, 3]

result_2 = list(it)
# => []
```

2回目の実行では、空のリストが返るようになりました。
これは、`filter`関数はイテレータを返し、列挙された要素は消費されてしまうためです。

`pnq`では、イテレータを暴露させずイテラブルとして動作し、イテレーション要求時に毎回新たなイテレータを生成するため、常に同じ結果を返します。

``` python
it = pnq.query([1, 2, 3]).filter(lambda x: True)

result_1 = list(it)
# => [1, 2, 3]

result_2 = list(it)
# => [1, 2, 3]
```

ただし、ソースがイテレータの場合はソースの挙動に準拠します。

``` python
it = pnq.query(filter(lambda x: True, [1, 2, 3]))

result_1 = list(it)
# => [1, 2, 3]

result_2 = list(it)
# => []
```

## 遅延評価・即時評価



## キャッシュ

## 非同期処理

### 非同期クエリの実行

非同期関数を含むクエリを実行するには`request`を使用する必要があり、また、クエリを非同期でイテレート（`async for ...`）する必要があります。

``` python
import asyncio

async def sleep(seconds):
    await asyncio.sleep(seconds)
    return seconds

async def main():
    async for x in pnq.infinite(lambda: 5).take(5).request(sleep):
        print(x)

asyncio.run(main())

```

### 同期クエリの実行

クエリは`__iter__`と`__aiter__`を備え、同期コードと非同期コードの互換性を最大限に保っています。`request`で実行する関数が非同期でない場合、いつも通り`for`を利用することで結果が得られます。`request`で実行する関数が非同期の場合は、同期クエリで実行できません。

``` python
import time

def sleep(seconds):
    time.sleep(seconds)
    return seconds

def main():
    for x in pnq.infinite(lambda: 5).take(5).request(sleep):
        print(x)

main()

```

### スリープ

`pnq`は`sleep`メソッドを提供しています。`sleep`は、実行形式（for/async for）に応じて、`time.sleep`と`asyncio.sleep`を使い分けます。
`sleep`は、指定した秒数だけ処理を中断した後、シーケンスから受け取った値を後続のチェインに受け流します。

``` python
def main():
    for x in pnq.infinite(lambda: 5).take(5).sleep(1).request(lambda x: x):
        print(x)

main()

```


### 非同期クエリのキャンセル管理

`pnq`は非同期処理をキャンセル要求から保護しません。
すなわち、システムがキャンセル要求を検知すると、処理は次回のawaitのタイミングでキャンセルされ、不完全な状態を引き起こす可能性があることに注意してください。

処理がキャンセルされるのを防ぐには、`pnq`の外で何らかのキャンセル管理を行う必要があります。
ひとつの方法として、`asyncio.shield`でコルーチンを保護します。

しかし、今度は逆にキャンセルができないという新たな問題を引き起こします。
非同期のキャンセル管理は難しく、何か非同期処理フレームワークを利用することをオススメします。

``` python
import asyncio

async def sleep(seconds):
    await asyncio.sleep(seconds)
    yield seconds

async def main():
    async for x in pnq.infinite(lambda: 5).take(5).request(sleep):
        print(x)

asyncio.run(asyncio.shield(main()))

```

### 例外処理

`request`は、実行結果を`params` `err` `result` `detail`のタプルとして受け渡します。
`err`が`None`でない場合、処理が失敗したとみなすことができ、次のようにトレースバックを取得できます。

``` python
import traceback

for params, err, result, detail in some_requests:
    if err:
        msg = "".join(
            traceback.format_exception(etype=type(err), value=err, tb=err.__traceback__)
        )
        # raise Exception("err") from err
        print(msg)
    else:
        print(result)
```

残っているリクエストを実行したくない場合は、次のようなコードを書いてもいいでしょう。

``` python
import traceback

for params, err, result, detail in some_requests:
    if err:
        raise RuntimeError(f"RequestError: {params} => {err}") from err
    else:
        print(result)
```


