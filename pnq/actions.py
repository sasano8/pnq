from typing import Mapping

__iter = iter
__next = next
__map = map
__filter = filter
__range = range
__all = all
__any = any
__max = max
__min = min
__sum = sum


###########################################
# iterating
###########################################
def iter(target):
    if hasattr(target, "__piter__"):
        yield from target.__piter__()
    elif isinstance(target, Mapping):
        yield from iter(target.items())
    else:
        yield from iter(target)


def infinite(func, *args, **kwargs):
    while True:
        yield func(*args, **kwargs)


def count(start=0, step=1):
    from itertools import count

    yield from count(start, step)


def cycle(iterable, repeat=None):
    from itertools import cycle

    yield from cycle(iterable)


###########################################
# mapping
###########################################
def map(self, func):
    return __map(func, self)


###########################################
# filtering
###########################################
def filter(self, func):
    return __filter(func, self)


###########################################
# partitioning
###########################################

###########################################
# aggregating
###########################################
def all(self):
    pass
