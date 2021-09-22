## 非同期処理

### 実行

次の場合、クエリは非同期の文脈で実行する必要があります。

- 非同期イテレータをソースとした場合
- `request_async`でクエリをチェインした場合
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
- `request_async`

これらのクエリは、実行結果を含んだオブジェクト`Response`を返し、
`Response`の`err`が`None`でない場合、処理は失敗したとみなせます。

処理が成功した場合、結果は`result`に格納されます。

スタックトレースは次のように取得できます。

``` python
import traceback

async def raise_error(x):
    raise Exception(str(x))

for res in pnq.query([{"x": 1}].request_async(raise_error)):
    if res.err:
        msg = "".join(
            traceback.format_exception(etype=type(err), value=err, tb=err.__traceback__)
        )
        print(msg)
    else:
        print(res.result)
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



