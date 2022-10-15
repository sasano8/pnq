# データソース

一般的に利用頻度が高いデータソースに対するクエリのショートカットを提供します。

※　本機能は次回リリースで提供されます。

# filesystem

ファイルやディレクトリに関するクエリを提供します。

::: pnq.ds.filesystem.ls
    :docstring:

::: pnq.ds.filesystem.files
    :docstring:

::: pnq.ds.filesystem.dirs
    :docstring:

# schedule

計画した間隔でイベント（`datetime.datetime(UTC)`）を提供し続けます。
無限イテレータのため、必要に応じて`take`等で終了条件を定めてください。

CPUの実行状況により、時間通りにイベントが送信されるとは限りません。
多少遅延することを想定してください。

::: pnq.ds.schedule.tick
    :docstring:

::: pnq.ds.schedule.tick_async
    :docstring:
