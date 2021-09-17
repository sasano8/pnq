<h1 align="center" style="font-size: 3rem; margin: -15px 0">
PNQ
</h1>

---

<div align="center">
<p>
<a href="https://github.com/sasano8/pnq/actions">
    <img src="https://github.com/sasano8/pnq/actions/workflows/test.yml/badge.svg" alt="Test Suite">
</a>
<a href="https://pypi.org/project/pnq/">
    <img src="https://badge.fury.io/py/pnq.svg" alt="Package version">
</a>
</p>

<em>User-friendly collection manipulation library.</em>
</div>

PNQ is a Python implementation like Language Integrated Query (LINQ).

https://pypi.org/project/pnq/

!!! danger
    PNQはベータ版です。

    - 現在、ドキュメントとAPIが一致していません。
    - PNQは鋭意開発中でAPIが頻繁に変更される恐れがあるため、本番環境では利用しないでください。

---



## Features

- コレクション操作に関する多彩な操作
- アクセシブルなインターフェース
- 型ヒントの活用
- 非同期ストリームに対応

## Documentation



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

pnq.query([1]).map(lambda x: x * 2).to_list()
# >> [2]

pnq.query({"a": 1, "b": 2}).filter(lambda x: x[0] == "a").to_list()
# >> [("a", 1)]

```
