"""
Copyright (c) 2021 sasano8

Released under the MIT license.
see https://opensource.org/licenses/MIT

The universal_write_open function from:
https://github.com/EntilZha/PyFunctional/blob/master/functional/io.py
https://github.com/EntilZha/PyFunctional/blob/master/functional/pipeline.py
"""

import builtins
import bz2
import gzip
import lzma


def universal_write_open(
    path,
    mode,
    buffering=-1,
    encoding=None,
    errors=None,
    newline=None,
    compresslevel=9,
    format=None,
    check=-1,
    preset=None,
    filters=None,
    compression=None,
):
    # pylint: disable=unexpected-keyword-arg,no-member
    if compression is None:
        return builtins.open(
            path,
            mode=mode,
            buffering=buffering,
            encoding=encoding,
            errors=errors,
            newline=newline,
        )
    elif compression in ("gz", "gzip"):
        return gzip.open(
            path,
            mode=mode,
            compresslevel=compresslevel,
            errors=errors,
            newline=newline,
            encoding=encoding,
        )
    elif compression in ("lzma", "xz"):
        return lzma.open(
            path,
            mode=mode,
            format=format,
            check=check,
            preset=preset,
            filters=filters,
            encoding=encoding,
            errors=errors,
            newline=newline,
        )
    elif compression == "bz2":
        return bz2.open(
            path,
            mode=mode,
            compresslevel=compresslevel,
            encoding=encoding,
            errors=errors,
            newline=newline,
        )
    else:
        raise ValueError(
            "compression must be None, gz, gzip, lzma, or xz and was {0}".format(
                compression
            )
        )
