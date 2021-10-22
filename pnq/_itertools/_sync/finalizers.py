import asyncio
from typing import List

from .._sync_generate.finalizers import (  # _all,; _any,; _sum,; each,
    _len,
    _max,
    _min,
    average,
    concat,
    contains,
    exists,
    first,
    first_or,
    first_or_raise,
    last,
    last_or,
    last_or_raise,
    one,
    one_or,
    one_or_raise,
    reduce,
)
from ..common import Listable, name_as


@name_as("all")
def _all(source, selector=None) -> bool:
    return all(Listable(source, selector))


@name_as("any")
def _any(source, selector=None) -> bool:
    return any(Listable(source, selector))


@name_as("sum")
def _sum(source, selector=None):
    return sum(Listable(source, selector))


def each(source, func=lambda x: x, unpack=""):
    if asyncio.iscoroutinefunction(func):
        raise TypeError("func must be a regular function")

    for elm in source:
        func(elm)
