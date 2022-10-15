from asyncio import sleep as asleep
from datetime import datetime
from functools import partial
from time import sleep

from pnq import query


def tick(seconds: float):
    def infinity(seconds):
        while True:
            yield datetime.utcnow()
            sleep(seconds)

    return query(infinity(seconds))


def tick_async(seconds: float):
    async def infinity(seconds):
        while True:
            yield datetime.utcnow()
            await asleep(seconds)

    return query(infinity(seconds))
