import asyncio
from functools import lru_cache, partial


@lru_cache(32)
def is_coroutine_function(func):
    # TODO: python3.8からはpartialが自動でasync functionを認識するので削除する
    if isinstance(func, partial):
        target = func.func
    else:
        target = func

    return asyncio.iscoroutinefunction(target)
