import traceback
from datetime import datetime, timezone
from json import dumps as _dumps
from typing import Any, Dict, NamedTuple, Tuple, Union
from threading import Event


def dumps(obj: Any) -> str:
    return _dumps(obj, ensure_ascii=False)


# TODO: is_cancelled はスペルが長くし、否定形で判定することが多いので嫌いだ。is_activeを採用したい。
class CancelToken:
    def __init__(self):
        self._is_active = True

    @property
    def is_cancelled(self):
        return not self._is_active

    @property
    def is_active(self):
        return self._is_active

    def cancel(self):
        self._is_active = False


# TODO: 未テスト
class ChainTokenBase(Event):
    def __init__(self, parent: "ChainTokenBase" = None):
        super().__init__()
        self._parent = parent
        self._reason = ""

        if parent:
            if not isinstance(parent, ChainTokenBase):
                raise TypeError(parent)

    def set(self, reason: str = "canceled"):
        with self._cond:
            if not self._flag:
                self._reason = str(reason)

            self._flag = True
            self._cond.notify_all()

    def is_set(self):
        if self.is_parent_set():
            return True
        else:
            return super().is_set()

    def is_parent_set(self):
        return self._parent and self._parent.is_set()

    def reason(self):
        if not self.is_set():
            raise RuntimeError("Not set.")
        else:
            if self._flag:
                return self._reason
            else:
                return self._parent.reason() + " by parent."

    def create_child(self):
        return ChainTokenBase(self)


class ChainToken(ChainTokenBase):
    def with_deadline(self, deadline: Union[datetime | str]):
        import time, threading, functools
        from time import sleep

        if isinstance(deadline, str):
            # deadline = "2018-01-02T03:04:05+09:00"
            deadline = datetime.fromisoformat(deadline)

        def handle_deadline(cancel_token: ChainTokenBase, deadline: datetime):
            deadline_ts = deadline.timestamp()
            while not cancel_token.is_set():
                sleep(1)
                now_ts = time.time()
                if deadline_ts < now_ts:
                    break
            cancel_token.set(f"cancelled by deadline => {deadline}")

        handler = functools.partial(handle_deadline, self, deadline)
        t = threading.Thread(target=handler, daemon=True)
        t.start()


class Request:
    args: Tuple
    kwargs: Dict


class Response(NamedTuple):
    func: Any
    args: Tuple
    kwargs: Dict
    err: Union[Exception, None]
    res: Any
    start: datetime
    end: datetime

    def result(self, timeout=None):
        if self.err:
            raise self.err
        else:
            return self.res

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

    def __str__(self):
        return str(self.to_dict())

    def to_dict(self, stack_trace: bool = True):
        """jsonに近いように辞書化します。kwargsとresultは解析されません。"""
        st = None
        err = None
        msg = None
        if self.err:
            err = self.err.__class__.__name__
            msg = str(self.err)
            if stack_trace:
                st = self.stack_trace

        return {
            "func": self.func.__name__,
            "args": self.args,
            "kwargs": self.kwargs,
            "result": self.res,
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "err": err,
            "msg": msg,
            "stack_trace": st,
        }

    def to_json(self, serializer=dumps):
        return serializer(self.to_dict())


class StopWatch:
    """コンテキスト内の処理時間を計測します。

    Args:

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
