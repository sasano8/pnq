from typing import Any, Iterable, Mapping, NoReturn, overload

from typing_extensions import Literal, Protocol


class PArgment(Protocol):
    args: Iterable
    kwargs: Mapping
