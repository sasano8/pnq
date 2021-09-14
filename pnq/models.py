import traceback
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, NamedTuple, Tuple, Union


class Request:
    args: Tuple
    kwargs: Dict


class Response(NamedTuple):
    func: Any
    args: Tuple
    kwargs: Dict
    err: Union[Exception, None]
    result: Any
    start: datetime
    end: datetime

    @property
    def elapsed(self):
        return self.end - self.start

    @property
    def stack_trace(self):
        err = self.err
        if not err:
            return ""
        return "".join(
            traceback.format_exception(etype=type(err), value=err, tb=err.__traceback__)
        )

    def to_dict(self):
        return {
            "func": self.func,
            "args": self.args,
            "kwargs": self.kwargs,
            "err": self.err,
            "result": self.result,
            "start": self.start,
            "end": self.end,
        }


class StopWatch:
    """コンテキスト内の処理時間を計測します。

    Parameters:

    * name: 任意の名前を付与できます

    Members:

    * name: 初期化時に付与した名前
    * start: コンテキストの開始時間（UTC）
    * end: コンテキストの完了時間（UTC）
    * elapsed: 開始時間と完了時間の差分秒数

    Usage:
    ```
    >>> with StopWatch("test") as result:
    >>>   [x for x in range(10000)]
    >>> print(result)
    {'name': 'test', 'start': '2021-09-13T14:10:04.780085+00:00', 'end': '2021-09-13T14:10:11.907716+00:00', 'elapsed': 7.127631}
    ```
    """

    name: str
    start: datetime
    end: datetime

    def __init__(self, name=""):
        self.name = name or ""
        self.start = None
        self.end = None

    def __enter__(self):
        if self.start:
            raise RuntimeError("StopWatch already started")
        start = datetime.utcnow()
        self.start = start.astimezone(timezone.utc)
        return self

    def __exit__(self, exc_value, exc_type, exc_tb):
        end = datetime.utcnow()
        self.end = end.astimezone(timezone.utc)

    def __str__(self):
        return str(self.to_dict())

    @property
    def elapsed(self):
        return (self.end - self.start).total_seconds()

    def to_dict(self):
        """計測データを辞書化します。日付データはisoformatで出力されます"""
        return {
            "name": self.name,
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "elapsed": self.elapsed,
        }
