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
