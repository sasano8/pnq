import asyncio
from functools import wraps


def to_sync(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        asyncio.run(func(*args, **kwargs))

    return wrapper
