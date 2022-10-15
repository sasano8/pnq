# PNQ

[![CI](https://github.com/sasano8/pnq/actions/workflows/test.yml/badge.svg)](https://github.com/sasano8/pnq/actions)
[![pypi](https://img.shields.io/pypi/v/pnq.svg)](https://pypi.python.org/pypi/pnq)
[![Downloads](https://pepy.tech/badge/pnq/month)](https://pepy.tech/project/pnq)

PNQ is a Python implementation like Language Integrated Query (LINQ).

!!! danger
    PNQはベータ版です。

    - 現在、ドキュメントとAPIが一致していません。
    - ライブラリが十分な品質に到達するまで、頻繁に内部実装やAPIが更新される恐れがあります。
    - 本番環境では利用しないでください。

---


## Features

- コレクション操作に関する多彩な操作
- アクセシブルなインタフェース
- 型ヒントの活用
- 非同期ストリームに対応

## Similar tools

- [PyFunctional](https://github.com/EntilZha/PyFunctional)
- [linqit](https://github.com/avilum/linqit)
- [python-linq](https://github.com/jakkes/python-linq)
- [aioitertools](https://github.com/omnilib/aioitertools)
- [asyncstdlib](https://github.com/maxfischer2781/asyncstdlib)
- [asq](https://github.com/sixty-north/asq)

## Documentation

- See [documentation](https://sasano8.github.io/pnq/) for more details.

## Dependencies

- Python 3.8+

## Installation

Install with pip:

```shell
$ pip install pnq
```

## Getting Started

``` python
import pnq

for x in pnq.query([1, 2, 3]).map(lambda x: x * 2):
    print(x)
# => 2, 4, 6

pnq.query([1, 2, 3]).map(lambda x: x * 2).result()
# => [2, 4, 6]

pnq.query([1, 2, 3]).filter(lambda x: x == 3).one()
# => 2
```

``` python
import asyncio
import pnq

async def aiter():
    yield 1
    yield 2
    yield 3

async def main():
    async for x in pnq.query(aiter()).map(lambda x: x * 2):
        print(x)
    # => 2, 4, 6

    await pnq.query(aiter()).map(lambda x: x * 2)
    # => [2, 4, 6]

    await pnq.query(aiter()).filter(lambda x: x == 3)._.one()
    # => 3

asyncio.run(main())
```

## release note

### v0.0.1 (2021-xx-xx)

* Initial release.
