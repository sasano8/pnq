<p align="center" style="margin: 0 0 10px">
  <img width="350" height="208" src="https://raw.githubusercontent.com/encode/httpx/master/docs/img/butterfly.png" alt='HTTPX'>
</p>

<h1 align="center" style="font-size: 3rem; margin: -15px 0">
HTTPX
</h1>

---

<div align="center">
<p>
<a href="https://github.com/encode/httpx/actions">
    <img src="https://github.com/encode/httpx/workflows/Test%20Suite/badge.svg" alt="Test Suite">
</a>
<a href="https://pypi.org/project/httpx/">
    <img src="https://badge.fury.io/py/httpx.svg" alt="Package version">
</a>
</p>

<em>A next-generation HTTP client for Python.</em>
</div>

HTTPX is a fully featured HTTP client for Python 3, which provides sync and async APIs, and support for both HTTP/1.1 and HTTP/2.


!!! note
    HTTPX should currently be considered in beta.

    We believe we've got the public API to a stable point now, but would strongly recommend pinning your dependencies to the `0.18.*` release, so that you're able to properly review [API changes between package updates](https://github.com/encode/httpx/blob/master/CHANGELOG.md).

    A 1.0 release is expected to be issued sometime in 2021.

---



## Features

- httpxのapiに基づいたリクエストの一括管理
- レスポンスの検証
- テストとバッチ処理支援

## Documentation



## Dependencies

- `httpx`

## Installation

Install with pip:

```shell
$ pip install httpx
```

Or, to include the optional HTTP/2 support, use:

```shell
$ pip install httpx[http2]
```

To include the optional brotli decoder support, use:

```shell
$ pip install httpx[brotli]
```

HTTPX requires Python 3.6+

[sync-support]: https://github.com/encode/httpx/issues/572


# Getting Started

```python
import mink

jobs = mink.load("sample.yml")
jobs = jobs.filter(tags=["production", "daily"], name=lambda x: "" in x)
jobs = jobs.tasks.filter(tags={"15"})

jobs.run(manager)

```


# expression

- 環境変数が読み込めます　例：　${ENV_NAME}
- 環境変数が存在しない場合のデフォルト値を設定できます　例：　${ENV_NAME:DEFAULT_VALUE}
- evalによる複雑な評価を行えます　例：　status_code: !eval 200 <= x and x < 300

# 設計

- 独自タグ
    - yamlのタグ機能で基本型以外も値として定義できます
    - ref:  <module>:<attr> | <variable> でpythonオブジェクトを参照することができます
    - type:  <module>:<attr> | <variable>  でpythonクラスを指定し初期化できます
    - eval: 式を評価することができます。

- リゾルバ
    - 環境変数: yamlのパース時に文字列として埋め込まれます。存在しない場合はNoneと解釈されます
    - 変数: yamlのアンカー等を利用します


# 類似ツール

## github actions
ifでevalを扱うことができる

``` yaml
    if: ${{ secrets.ref == 'refs/heads/main' }}
```

github actionsはsecretsと呼ばれるコンテキストに変数を設定し、secretsで変数にアクセスする。
yamlに展開すると、機密情報がログに出力されてしまったりするので、サンドボックスでのみアクセス可能にするのはいい考えのような気がする

## tavern
saveで変数を保持し、"{val_name:type}"で参照できる

``` yaml
stages:
  - name: Make sure we have the right method
    request:
      url: https://kaka-request-dumper.herokuapp.com/
      method: GET
      save:
        json:
          returned_method: method

  - name: Make sure we have the right returned_method
    request:
      url: https://kaka-request-dumper.herokuapp.com/
      method: GET
      params:
        returned_method: "{returned_method:s}"
```

includeで共通定義を読み込める


```
description: used for github api testing
name: test includes
variables:
  service:
    token: "token"
    owner: "リポジトリオーナー"
    repo: "リポジトリ"
```

```
includes:
  - !include common.yaml
```

環境変数展開については記述がない
pytestで実行でき、pytestで結果を表示できる




``` mermaid
graph TD;
    A-->B;
    A-->C;
    B-->D;
    C-->D;
```


``` mermaid
sequenceDiagram
  ユーザ    ->> +Vue         : ログインボタンクリック
  Vue      ->> +Laravel     : ログインAPI
  Laravel  ->> +Database    : SQL
    Note right of Database  : 認証テーブル参照
  Database ->> -Laravel     : Result
  alt ログイン成功
    Laravel ->> Vue         : success
  else 失敗
    Laravel ->> -Vue        : failure
  end
  Vue       ->> -ユーザ      : 結果表示
```

``` mermaid
graph TB
  Macの選び方 --> 持ち歩く
  持ち歩く -->|はい| スペック
  持ち歩く -->|いいえ| 予算
  スペック -->|必要| R1[MacBook Pro]
  スペック -->|低くても良い| R2[MacBook Air]
  予算 --> |いくらでもある| R3[Mac Pro]
  予算 --> |できれば抑えたい| R4[Mac mini / iMac]
```

``` mermaid
classDiagram
  class Loader {
    -int HP
  }
  class Schema {
    -int HP
  }
  class Parser {
    -parse_file()
  }
  class Composer {
    -parse_file()
  }
  class Context {
    -int HP
  }
  class Sandbox {
    -ファイア()
  }
  Composer <|-- Context
  Composer <|-- Sandbox
  Composer <|-- Parser
  Parser <|-- Loader
  Parser <|-- Schema
```



``` mermaid
gantt
  title PHP life cycle

  section PHP 7.2
    active support      : done, 2017-11-30, 2019-11-30
    security support     : crit, 2020-11-30

  section PHP7.3
    active support      : 2018-12-06, 2020-12-06
    security support     : crit, 2021-12-06

  section PHP7.4
    active support      : 2019-11-28, 2021-11-28
    security support     : crit, 2022-11-28

```

``` mermaid
pie
  "iOS": 45.2
  "iPhone": 17.2
  "PHP": 8.6
  "Objective-C": 6.5
  "Swift": 6.5
  "Xcode": 4
  "Laravel": 3
  "Realm": 3
  "Android": 3
  "Others": 2
```


``` mermaid
graph TB
    subgraph s1
        s1a --> s1b
    end
    subgraph s2
        s2a --> s2b
        s2a --> s1b
    end
```

