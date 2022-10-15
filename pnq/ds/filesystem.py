from glob import iglob
from os.path import isfile
from pathlib import Path

from pnq import query


def ls(pathname="*", *, root_dir=None, dir_fd=None, recursive=False):
    return query(
        Path(x)
        for x in iglob(
            pathname=pathname, root_dir=root_dir, dir_fd=dir_fd, recursive=recursive
        )
    )


def files(pathname="*", *, root_dir=None, dir_fd=None, recursive=False):
    return query(
        Path(x)
        for x in iglob(
            pathname=pathname, root_dir=root_dir, dir_fd=dir_fd, recursive=recursive
        )
        if isfile(x)
    )


def dirs(pathname="*", *, root_dir=None, dir_fd=None, recursive=False):
    return query(
        Path(x)
        for x in iglob(
            pathname=pathname, root_dir=root_dir, dir_fd=dir_fd, recursive=recursive
        )
        if not isfile(x)
    )
