from typing import (
    Any,
    ItemsView,
    KeysView,
    Protocol,
    TypeVar,
    ValuesView,
    runtime_checkable,
)

undefined: Any = object()

K = TypeVar("K", covariant=True)
V = TypeVar("V", covariant=True)


@runtime_checkable
class KeyValueItems(Protocol[K, V]):
    def keys(self) -> KeysView[K]:
        return self.source.keys()  # type: ignore

    def values(self) -> ValuesView[V]:
        return self.source.values()  # type: ignore

    def items(self) -> ItemsView[K, V]:
        return self.source.items()  # type: ignore


KeyValueItems.register(dict)
