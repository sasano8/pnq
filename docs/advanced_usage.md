# 高度な使用方法

## 並列処理

`pnq`は任意のエクゼキュータでの並列処理をサポートし、効率的に計算資源を活用できます。
並列処理に対応しているメソッドは次の通りです。

- parallel（結果のみを返す）
- request（成功・失敗情報を含む結果を返す）
- dispatch（処理を投げっぱなしにする）

また、`gather`を使うと複数のクエリを並列実行し、非同期に完了待機できます。
`gather`はコルーチンなど`awaitable`なオブジェクトに対応しています。

次の例は、クエリを並列実行し、実行中のクエリの完了を待たずに次のクエリを順次実行する例です。
ただし、むやみな並列処理はデッドロックやメモリ圧迫などの問題を引き起こすので控えてください。


``` python
import asyncio
import pnq
from pnq.concurrent import ProcessPool, ThreadPool, AsyncPool

def mul(x):
    return x * 2

async def mul_async(x):
    return x * 2

async def aiter():
    yield 1
    yield 2
    yield 3

async def notify(x):
    await asyncio.sleep(0.1)
    print(x)

async def main():
    async with ProcessPool(2) as proc, ThreadPool(2) as thread, AsyncPool(2) as aio:
        tasks = pnq.query([
            pnq.query([1, 2, 3]).parallel(mul, proc),
            pnq.query([1, 2, 3]).parallel(mul_async, proc),
            pnq.query(aiter()).parallel(mul, proc),
            pnq.query(aiter()).parallel(mul_async, proc),

            pnq.query([1, 2, 3]).parallel(mul, thread),
            pnq.query([1, 2, 3]).parallel(mul_async, thread),
            pnq.query(aiter()).parallel(mul, thread),
            pnq.query(aiter()).parallel(mul_async, thread),

            pnq.query([1, 2, 3]).parallel(mul, aio),
            pnq.query([1, 2, 3]).parallel(mul_async, aio),
            pnq.query(aiter()).parallel(mul, aio),
            pnq.query(aiter()).parallel(mul_async, aio),
        ])

        await tasks.gather().flat().dispatch(notify, aio)

asyncio.run(main())
```

`pnq`が提供するエクゼキュータは次の通りです。

### ProcessPool
CPUバウンドな重たい処理に向いています。GILの制限を受けません。
チャンクサイズを指定すると、一括処理を効率化できます。
チャンクサイズは`ProcessPool`でのみ有効で、それ以外のエクゼキューターでは無視されます。

- 同期関数と非同期関数はプロセスプール上で実行されます。

### ThreadPool
I/Oバウンドなプリエンプティブマルチタスク（time.sleepなど）に向いています。

- 同期関数と非同期関数はスレッドプール上で実行されます。

### AsyncPool
I/Oバウンドなノンエンプティブマルチタスク（asyncio.sleepなど）に向き、シングルスレッドを効率的に利用します。
同期関数はスレッドプール上で実行されるため、ThreadPoolの代用としても働きます。

- 同期関数はスレッドプール上で実行されます。
- 非同期関数はイベントループ上（asyncio）で実行されます。

クエリは非同期実行のみ許可されます。

### DummyPool
エクゼキュータを指定しない場合に使用されます。

- 同期実行時は並列化されず、単に現在のスレッドで処理を即時実行します。
- 非同期実行時は、`AsyncPool`のように振る舞います。
- 同時実行数は、指定した同時実行数＋1（同期実行は即時処理）となります。

コンテキストが存在せず、投げっぱなしにされた処理の実行は保証されないため、
基本的には別のエクゼキュータを指定するようにしてください。



## 非同期処理

### 実行

次の場合、クエリは非同期の文脈で実行する必要があります。

- 非同期イテレータをソースとした場合
- `request`による並列処理を非同期に待ち受けたい場合

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

    async for x in pnq.query([dict(x=1), dict(x=2), dict(x=3)]).request(sleep):
        print(x)


asyncio.run(main())
```


### キャンセル管理

`pnq`は簡単なキャンセル機構を提供し、これを利用できます。

`pnq.run`に、非同期関数を渡すと、その関数を起動し、第一引数にキャンセルトークンを渡します。
第一引数を受け入れ可能な場合、その関数はキャンセルコントロールの意思があるとみなされます。

次のコードは、10秒間待機している間キャンセル（SIGTERMとSIGINT）を受け入れません。

``` python
import asyncio
import pnq


async def main(token):
    await asyncio.sleep(10)
    print("Hello, world!")


pnq.run(main)
```

キャンセルを検知すると`token.is_running`は`False`を返すようになります。
`token.is_running`を監視することで、任意のタイミングで処理を中断できます。

次のコードは、`token.is_running`が`False`と評価されるまで、要素を流し続けます。

``` python
import asyncio
import pnq


async def main(token):
    async def infinity(token):
        while token.is_running:
            yield 1
            await asyncio.sleep(1)

    async for x in pnq.query(infinity(token)):
        print(x)


pnq.run(main)
```

関数が第一引数を受け入れ可能でない場合、その関数はキャンセルコントロールの意思がないとみなされます。

次のコードは、キャンセルを検知すると実行は即時キャンセルされます。

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

`pnq`は基本的に例外をキャッチしませんが、次のクエリのみ関数実行時の例外をキャッチします。

- `request`

これらのクエリは、実行結果を含んだオブジェクト`Response`を返し、
`Response`の`err`が`None`でない場合、処理は失敗したとみなせます。

結果は`result`から取得できます。
処理が失敗している場合は例外が返ります。

スタックトレースは次のように取得できます。

``` python
import traceback

async def raise_error(x):
    raise Exception(str(x))

for res in pnq.query([{"x": 1}].request(raise_error)):
    if res.err:
        msg = "".join(
            traceback.format_exception(etype=type(err), value=err, tb=err.__traceback__)
        )
        print(msg)
    else:
        print(res.result())
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

        async def fetch_from_url(url):
            res = await client.get(url)
            res.raise_for_status()
            return res

        @params.request(fetch_from_url, unpack="**").group_by
        def split_success_and_error(res):
            return (not res.err, res)

        return dict(await split_success_and_error)

result = asyncio.run(main())
# {
#   True: [res1, res2, ...],
#   False: [res3, res4, ...],
# }
```



