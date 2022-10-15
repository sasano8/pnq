import platform
from glob import iglob
from os.path import isfile
from pathlib import Path

from pnq import query

# TODO: 3.10以降では削除
V_3_10 = int(platform.python_version_tuple()[1]) >= 10


# TODO: drop from 3.10
def _build_args(**kwargs):
    if V_3_10:
        return kwargs
    else:
        del kwargs["root_dir"]
        del kwargs["dir_fd"]
        return kwargs


def pathlib_to_str(x):
    if isinstance(x, Path):
        return str(x)
    else:
        return x


def ls(pathname="*", *, root_dir=None, dir_fd=None, recursive=False):
    pathname = pathlib_to_str(pathname)
    kwargs = _build_args(
        pathname=pathname, root_dir=root_dir, dir_fd=dir_fd, recursive=recursive
    )
    return query(Path(x) for x in iglob(**kwargs))


def files(pathname="*", *, root_dir=None, dir_fd=None, recursive=False):
    pathname = pathlib_to_str(pathname)
    kwargs = _build_args(
        pathname=pathname, root_dir=root_dir, dir_fd=dir_fd, recursive=recursive
    )
    return query(Path(x) for x in iglob(**kwargs) if isfile(x))


def dirs(pathname="*", *, root_dir=None, dir_fd=None, recursive=False):
    pathname = pathlib_to_str(pathname)
    kwargs = _build_args(
        pathname=pathname, root_dir=root_dir, dir_fd=dir_fd, recursive=recursive
    )
    return query(Path(x) for x in iglob(**kwargs) if not isfile(x))
