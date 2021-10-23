from dataclasses import dataclass
from functools import partial
from typing import Tuple


class Undefined:
    def __eq__(self, o):
        raise AttributeError("undefined cannot be referenced.")

    def __ne__(self, o):
        raise AttributeError("undefined cannot be referenced.")

    def __bool__(self):
        raise AttributeError("undefined cannot be referenced.")


undefined: bool = Undefined()  # type: ignore


class TestException(Exception):
    ...


@dataclass
class ExecutorSpec:
    is_cpubound: bool = undefined
    max_workers: bool = undefined
    is_async_only: bool = undefined
    __enter__: bool = undefined
    __exit__: bool = undefined
    __aenter__: bool = undefined
    __aexit__: bool = undefined
    submit_sync: bool = undefined
    submit_async: bool = undefined
    asubmit_sync: bool = undefined
    asubmit_async: bool = undefined
    submit_partial_sync: bool = undefined
    submit_partial_async: bool = undefined
    asubmit_partial_sync: bool = undefined
    asubmit_partial_async: bool = undefined

    @classmethod
    async def test_capability(cls, factory, is_async_only, submit_sync=True):
        _is_async_only, spec = await test_executor_spec(factory)
        # required
        assert spec.is_cpubound
        assert spec.max_workers
        assert spec.is_async_only
        assert spec.__aenter__
        assert spec.__aexit__
        assert _is_async_only == is_async_only

        if _is_async_only:
            assert not spec.__enter__
            assert not spec.__exit__
            assert spec.asubmit_sync
            assert spec.asubmit_async
            if not submit_sync:
                assert isinstance(spec.submit_sync, NotImplementedError)
                assert isinstance(spec.submit_partial_sync, NotImplementedError)
            else:
                assert spec.submit_sync
            assert isinstance(spec.submit_async, NotImplementedError)
            assert isinstance(spec.submit_partial_async, NotImplementedError)

        else:
            assert submit_sync
            assert spec.__enter__
            assert spec.__exit__
            assert spec.submit_sync
            assert spec.submit_async
            assert spec.asubmit_sync
            assert spec.asubmit_async
            assert spec.submit_partial_sync
            assert spec.submit_partial_async
            assert spec.asubmit_partial_sync
            assert spec.asubmit_partial_async

        return True


def return_one():
    return 1


async def return_two():
    return 2


async def test_executor_spec(factory) -> Tuple[bool, ExecutorSpec]:
    spec = ExecutorSpec()

    try:
        with factory(1) as f:
            raise TestException()

    except TestException:
        spec.__enter__ = True
        spec.__exit__ = True

    except NotImplementedError:
        spec.__enter__ = False
        spec.__exit__ = False

    try:
        async with factory(1) as f:
            raise TestException()

    except TestException:
        spec.__aenter__ = True
        spec.__aexit__ = True

    async with factory(1) as executor:
        spec.is_cpubound = isinstance(executor.is_cpubound, bool)
        spec.max_workers = isinstance(executor.is_cpubound, int)
        spec.is_async_only = isinstance(executor.is_async_only, bool)
        spec.submit_sync = try_submit(executor, return_one, 1)
        spec.submit_async = try_submit(executor, return_two, 2)
        spec.asubmit_sync = await try_asubmit(executor, return_one, 1)
        spec.asubmit_async = await try_asubmit(executor, return_two, 2)

        # python3.8からはpartialはasync functionを認識できる模様
        spec.submit_partial_sync = try_submit(executor, partial(return_one), 1)
        spec.submit_partial_async = try_submit(executor, partial(return_two), 2)
        spec.asubmit_partial_sync = await try_asubmit(executor, partial(return_one), 1)
        spec.asubmit_partial_async = await try_asubmit(executor, partial(return_two), 2)

        return executor.is_async_only, spec


def try_submit(executor, func, expect):
    try:
        return executor.submit(func).result() == expect
    except Exception as e:
        return e


async def try_asubmit(executor, func, expect):
    try:
        return await executor.asubmit(func) == expect
    except Exception as e:
        return e
