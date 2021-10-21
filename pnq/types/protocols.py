from typing import Iterable, Mapping

from typing_extensions import Protocol


class PArgment(Protocol):
    args: Iterable
    kwargs: Mapping
