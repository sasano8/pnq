## 非同期処理

### 実行

次の場合、クエリは非同期の文脈で実行する必要があります。

- 非同期イテレータをソースとした場合
- `request_async`でクエリをチェーンした場合
- `each_async`を呼び出す場合

``` python
import asyncio
import pnq

async def async_iterate():
    yield 1
    yield 2
    yield 3

async def sleep(x):
    await asyncio.sleep(1)
    print(x)

async def main():
    async for x in pnq.query(async_iterate()):
        print(x)

    async for x in pnq.query([dict(x=1), dict(x=2), dict(x=3)]).request_async(sleep):
        print(x)

    await pnq.query([1, 2, 3]).each_async(sleep)

asyncio.run(main())
```


### 実装例

非同期なリクエストを含むクエリの実装例です。

``` python
import asyncio
import pnq
import httpx


async def main():
    async with httpx.AsyncClient() as client:
        params = pnq.query([
            {"url": "test_url_1"},
            {"url": "test_url_2"},
        ])

        @params.request_async
        async def fetch_from_url(url):
            res = await client.get(url)
            res.raise_for_status()
            return res

        @fetch_from_url.group_by
        def split_success_and_error(res):
            return (not res.err, res)

        return await split_success_and_error.lazy(dict)

result = asyncio.run(main())
# {
#   True: [res1, res2, ...],
#   False: [res3, res4, ...],
# }
```

### キャンセル管理

`pnq`は簡単なキャンセル機構を提供し、これを利用できます。

`pnq.run`に、非同期関数を渡すと、その関数を起動し、第一引数にキャンセルトークンを渡します。
第一引数を受け入れ可能な場合、その関数はキャンセルをコントロールする意思があるとみなされます。

次のコードは、10秒間待機している間キャンセル（SIGTERMとSIGINT）を受け入れません。

``` python
import asyncio
import pnq


async def main(token):
    await asyncio.sleep(10)
    print("Hello, world!")


pnq.run(main)
```

キャンセルを検知すると`token.is_running()`は`False`を返すようになります。
`token.is_running()`を監視することで、任意のタイミングで処理を中断できます。

次のコードは、`token.is_running()`が`False`と評価されるまで、要素を流し続けます。

``` python
import asyncio
import pnq


async def main(token):
    async def infinity():
        while True:
            yield 1
            await asyncio.sleep(1)

    async for x in pnq.query(infinity()).take_while(token.is_running):
        print(x)


pnq.run(main)
```

関数が第一引数を受け入れ可能でない場合、実行は即時にキャンセルされます。

``` python
import asyncio
import pnq


async def main():
    async def infinity():
        while True:
            yield 1
            await asyncio.sleep(1)

    async for x in pnq.query(infinity()):
        print(x)


pnq.run(main)
```


### 例外処理

`request_async`は、例外を補足し実行結果を含んだオブジェクト`Response`を返します。
`Response`オブジェクトの`err`が`None`でない場合、処理が失敗したとみなすことができ、次のようにスタックトレースを取得できます。

``` python
import traceback

for res in some_requests:
    if res.err:
        msg = "".join(
            traceback.format_exception(etype=type(err), value=err, tb=err.__traceback__)
        )
        # raise Exception("err") from err
        print(msg)
    else:
        print(result)
```




## 性能評価

### pnqのイテレーション性能

内包表記と比較すると`pnq`は1.36程度性能が落ちます。
日常的に大量のデータを処理する場合は、ネイティブな記法に書き直すことも検討ください。

``` python
import pnq
from pnq.base.requests import StopWatch
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

```
{'name': '内包表記', 'start': '2021-09-20T12:42:35.219151+00:00', 'end': '2021-09-20T12:42:42.270021+00:00', 'elapsed': 7.05087}
{'name': 'イテレータ', 'start': '2021-09-20T12:42:42.270046+00:00', 'end': '2021-09-20T12:42:49.383548+00:00', 'elapsed': 7.113502}
{'name': 'pnq', 'start': '2021-09-20T12:42:49.383573+00:00', 'end': '2021-09-20T12:42:58.979269+00:00', 'elapsed': 9.595696}
内包表記:pnq 性能差割 ：1.36
```

### 非同期イテレータの性能

非同期イテレータは同期イテレータより性能が2.25倍程度遅いです。

非同期イテレータは、ネットワークI/OやファイルI/Oなどの待機時間で、
並列処理できるケースで有効です。

特に理由がない場合は、同期イテレータを積極的に使うようにしてください。

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
    print(f"性能差割合　　：{rate}")


asyncio.run(main())
```

```
同期　　　　　：{'start': '2021-09-13T10:28:55.240113+00:00', 'end': '2021-09-13T10:28:58.890342+00:00', 'elapsed': 3.650229}
非同期　　　　：{'start': '2021-09-13T10:28:58.890577+00:00', 'end': '2021-09-13T10:29:07.085747+00:00', 'elapsed': 8.19517}
性能劣化率　　：2.25
```


