# 性能評価

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


