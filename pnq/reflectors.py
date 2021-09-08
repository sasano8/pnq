from dataclasses import asdict, is_dataclass
from functools import partial


def return_as_it_is(v):
    return v


def to_str(v):
    if v is None:
        return ""
    else:
        return str(v)


def as_dict(v):
    converter = get_asdict(v)
    if converter:
        return converter()
    else:
        raise TypeError(f"{v} is not a supported asdict.")


def try_as_dict(v):
    converter = get_asdict(v)
    if converter:
        return converter()
    else:
        return v


def get_asdict(v):
    if to_dict := is_namedtuple(v):
        return to_dict
    elif to_dict := is_pydantic(v):
        return to_dict
    elif is_dataclass(v):
        return partial(asdict, v)
    elif isinstance(v, dict):
        return partial(return_as_it_is, v)
    else:
        return None


def is_namedtuple(v):
    if isinstance(v, tuple) and hasattr(v, "_asdict"):
        return v._asdict  # type: ignore
    else:
        return None


try:
    from pydantic import BaseModel

    def is_pydantic(v):
        if isinstance(v, BaseModel):
            return v.dict
        else:
            return None


except ImportError:

    def get_asdict(v):
        if is_namedtuple(v):
            return v.asdict
        elif is_dataclass(v):
            return partial(asdict, v)
        else:
            return False
