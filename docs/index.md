# PNQ

[![CI](https://github.com/sasano8/pnq/actions/workflows/test.yml/badge.svg)](https://github.com/sasano8/pnq/actions)
[![pypi](https://img.shields.io/pypi/v/pnq.svg)](https://pypi.python.org/pypi/pnq)
[![downloads](https://pepy.tech/badge/pydantic/month)](https://pepy.tech/project/pydantic)

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

## Documentation

- See [documentation](https://sasano8.github.io/pnq/) for more details.

## Dependencies

- Python 3.7+

## Installation

Install with pip:

```shell
$ pip install pnq
```

## Getting Started

```python
import pnq

pnq.query([1]).map(lambda x: x * 2).to(list)
# >> [2]

pnq.query({"a": 1, "b": 2}).filter(lambda x: x[0] == "a").to(list)
# >> [("a", 1)]

```
