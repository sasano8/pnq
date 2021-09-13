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

## 遅延評価・即時評価・キャッシュ



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


### 実装例

``` python
import asyncio
import pnq
import httpx

SUCCESS = True
ERROR = False

params = pnq.query([
    {"url": "test_url_1"},
    {"url": "test_url_2"},
])

@params.request
async def step_fetch(url):
    async with httpx.AsyncClient() as client:
        res = await client.get("url")
        res.raise_for_status()
        return res

@step_fetch.group_by
def step_finalize(res):
    return ERROR if res.err else SUCCESS, res

result = asyncio.run(step_finalize.lazy(dict))
print(result[SUCCESS])
print(result[ERROR])
```



## 性能評価

### pnqのイテレーション性能

`pnq`は基本的にPython標準の書き方より性能が遅くなります。
処理の汎用化を図るためのオーバーヘッドが生じているためです。

例を見てみましょう。

``` python
import pnq
from pnq.models import StopWatch
from decimal import Decimal, ROUND_HALF_UP


RANGE = 100000000


def dummy(x):
    return x


with StopWatch("内包表記") as result_1:
    list(dummy(x) for x in range(RANGE) if x % 2)


with StopWatch("イテレータ") as result_2:

    def iterate():
        for i in range(RANGE):
            if i % 2:
                yield dummy(i)

    list(iterate())


with StopWatch("pnq") as result_3:
    pnq.query(range(RANGE)).filter(lambda x: x % 2).map(dummy).to_list()

difference = Decimal(f"{result_1.elapsed}") - Decimal(f"{result_3.elapsed}")
rate = Decimal(f"{result_3.elapsed}") / Decimal(f"{result_1.elapsed}")
rate = rate.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

print(f"{result_1}")
print(f"{result_2}")
print(f"{result_3}")
print(f"内包表記:pnq 性能差割 ：{rate}")
```

一般的に、内包表記が最も早くなります。
`pnq`は性能では太刀打ちできないでしょう。

```
{'name': '内包表記', 'start': '2021-09-13T14:10:04.780085+00:00', 'end': '2021-09-13T14:10:11.907716+00:00', 'elapsed': 7.127631}
{'name': 'イテレータ', 'start': '2021-09-13T14:10:11.907728+00:00', 'end': '2021-09-13T14:10:19.175027+00:00', 'elapsed': 7.267299}
{'name': 'pnq', 'start': '2021-09-13T14:10:19.175040+00:00', 'end': '2021-09-13T14:10:30.971770+00:00', 'elapsed': 11.79673}
内包表記:pnq 性能差 ：1.66
```

### 非同期イテレータの性能

非同期イテレータは、同期イテレータより一般的に遅いです。
例を見てみましょう。

``` python
import asyncio
from decimal import Decimal, ROUND_HALF_UP

from pnq.models import StopWatch


class Range:
    def __init__(self, count):
        self.count = count

    def __iter__(self):
        for i in range(self.count):
            yield i

    async def __aiter__(self):
        for i in range(self.count):
            yield i


calculator = Range(100000000)


async def main():
    with StopWatch() as result_1:
        for i in calculator:
            pass

    with StopWatch() as result_2:
        async for i in calculator:
            pass

    difference = Decimal(f"{result_1.elapsed}") - Decimal(f"{result_2.elapsed}")
    rate = Decimal(f"{result_2.elapsed}") / Decimal(f"{result_1.elapsed}")
    rate = rate.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

    print(f"同期　　　　　：{result_1}")
    print(f"非同期　　　　：{result_2}")
    print(f"同期 - 非同期 ：{difference}")
    print(f"性能差割合　　：{rate}")


asyncio.run(main())
```

非同期イテレータは同期イテレータより性能が2.25倍程度遅くなりました。

```
同期　　　　　：{'start': '2021-09-13T10:28:55.240113+00:00', 'end': '2021-09-13T10:28:58.890342+00:00', 'elapsed': 3.650229}
非同期　　　　：{'start': '2021-09-13T10:28:58.890577+00:00', 'end': '2021-09-13T10:29:07.085747+00:00', 'elapsed': 8.19517}
同期 - 非同期 ：-4.544941
性能劣化率　　：2.25

```

非同期イテレータが効力を発揮するケースは、ネットワークI/OやファイルI/Oなど待機時間が多く、
その間に並列処理で時間を有効活用できる場合です。

特に理由がない場合は、同期イテレータを積極的に使うようにしてください。

