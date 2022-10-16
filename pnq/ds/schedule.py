from asyncio import sleep as asleep
from datetime import datetime
from time import sleep

from pnq import query
from pnq._itertools.requests import CancelToken


# TODO: 長いsecondsが与えられた時、その待機中トークンの判定がされない。sleepを細切れにして、できるだリアルタイムに判定されるようにする
def tick(seconds: float, token: CancelToken = None):
    token = token or CancelToken()

    def infinity(seconds, token: CancelToken):
        while token.is_active:
            yield datetime.utcnow()
            sleep(seconds)

    return query(infinity(seconds, token))


def tick_async(seconds: float, token: CancelToken = None):
    token = token or CancelToken()

    async def infinity(seconds, token: CancelToken):
        while token.is_active:
            yield datetime.utcnow()
            await asleep(seconds)

    return query(infinity(seconds, token))


class timezones:
    @classmethod
    def keys(cls):
        import zoneinfo

        return query(zoneinfo.available_timezones())

    @classmethod
    def values(cls):
        from zoneinfo import ZoneInfo

        return query(ZoneInfo(x) for x in cls.keys())

    @classmethod
    def items(cls):
        from zoneinfo import ZoneInfo

        return query((x, ZoneInfo(x)) for x in cls.keys())


class weekdays:
    """datetime.weekday()で返される値の対応表"""

    @classmethod
    def keys(cls):
        return query(x[0] for x in cls.items())

    @classmethod
    def values(cls):
        return query(x[1] for x in cls.items())

    @classmethod
    def items(cls):
        return query(
            {
                0: "Monday",
                1: "Tuesday",
                2: "Wednesday",
                3: "Thursday",
                4: "Friday",
                5: "Saturday",
                6: "Sunday",
            }
        )


class isoweekdays(weekdays):
    """datetime.isoweekday()で返される値の対応表"""

    @classmethod
    def items(cls):
        return query(
            {
                1: "Monday",
                2: "Tuesday",
                3: "Wednesday",
                4: "Thursday",
                5: "Friday",
                6: "Saturday",
                7: "Sunday",
            }
        )


class spans:
    @classmethod
    def keys(cls):
        return query(x[0] for x in cls.items())

    @classmethod
    def values(cls):
        return query(x[1] for x in cls.items())

    @classmethod
    def items(cls):
        return query(
            {
                1: "year",
                2: "month",
                3: "day",
                4: "hour",
                5: "minute",
                6: "second",
                7: "microsecond",
            }
        )
