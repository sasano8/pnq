# 環境構築

```
pyenv local 3.8
```

``` shell
poetry install
poetry run pre-commit install
```

# フォーマット

```
poetry run black .
```

# 開発ガイド

1. actions.py: 新たに定義するクエリの説明を記述します。
2. docs/api.md: actions.pyへの参照をドキュメントに反映します。
3. `_async`に非同期クエリを実装します（まずは仮の実装でよいでしょう）。あるいは、`_sync`に同期クエリを実装します（非同期クエリから自動生成できる互換性がない場合）。
4. `queryables.py`に非同期クエリへの参照（`_ait`）と同期クエリ（`_sit`）への参照を定義します。実装がない場合は、`no_implement`とします。
5. `make generate` を実行します。実行前に、これまでの作業内容をステージングし、どんな変更があるか確認してください。
6. `tests`にテストを追加します。

<!--
TODO: １クエリを定義するのに手続きが多い。もっと簡略化したい。
TODO: actions.py が説明だけのモジュールなのでもっと一箇所に定義を集めたい
TODO: docs/api.md への繁栄を自動にしたい
-->


## 同期/非同期

同期クエリと非同期クエリをそれぞれ実装するコストが高いため、
非同期クエリを同期クエリへ変換するアプローチを取っています。

次のルールを参考にコードを実装します。

- `_async`: 非同期クエリを定義します。
- `_sync_generate`: 非同期クエリから自動生成される同期クエリ群。このモジュールは触らないでください。
- `_sync`: `_sync_generate`がインポートされる。自動生成できない、あるいは、最適化したいクエリをここでオーバーライドしてください。
- `generate_unasync.py`: 非同期から同期コードへの変換方法を定義する。このモジュールは触らないでください。

`generate_unasync.py` により、非同期コードは同期コードに変換されます。
`generate_unasync.py`には、次のようなルールが定義されます。

```
fromdir=str(CODE_ROOT / "_async"),
todir=str(CODE_ROOT / "_sync_generate"),
```

次を実行するとコードが自動生成されます。

```make unasync```

さらに詳しくは[unasync](https://github.com/python-trio/unasync)を参照してください。


## ジェネリクステンプレート

本パッケージのジェネリクスは複雑です。
正しく型を伝えるため、あるいは、曖昧な型を吸収するためのテンプレートです。

主に、`Tuple[KEY, VALUE]`型の活用範囲を広げることを目的としています。

- `__queries__.py`: Queryクラスのテンプレートです。クエリを追加したらここへコードを追加します。
- `queries.py`: `__queries__.py`をテンプレートとして生成されるモジュールです。

次を実行するとコードが自動生成されます。

```make generate```
