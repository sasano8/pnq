"""
Copyright (c) 2021 sasano8

Released under the MIT license.
see https://opensource.org/licenses/MIT

The universal_write_open function from:
https://github.com/EntilZha/PyFunctional/blob/master/functional/io.py
https://github.com/EntilZha/PyFunctional/blob/master/functional/pipeline.py
"""

import csv
import json
from functools import partial
from typing import Literal, Union

from ..io import universal_write_open


def to_file(
    iterable,
    path,
    delimiter=None,
    mode: Union[Literal["r", "w", "x", "a", "b", "t", "+"], str] = "wt",
    buffering=-1,
    encoding=None,
    errors=None,
    newline=None,
    compresslevel=9,
    format=None,
    check=-1,
    preset=None,
    filters=None,
    compression: Union[None, Literal["gz", "gzip", "lzma", "xz", "bz2"]] = None,
):
    """
    Saves the sequence to a file by executing str(self) which becomes str(self.to_list()). If
    delimiter is defined will instead execute self.make_string(delimiter)
    :param path: path to write file
    :param delimiter: if defined, will call make_string(delimiter) and save that to file.
    :param mode: file open mode
    :param buffering: passed to builtins.open
    :param encoding: passed to builtins.open
    :param errors: passed to builtins.open
    :param newline: passed to builtins.open
    :param compression: compression format
    :param compresslevel: passed to gzip.open
    :param format: passed to lzma.open
    :param check: passed to lzma.open
    :param preset: passed to lzma.open
    :param filters: passed to lzma.open
    """
    with universal_write_open(
        path,
        mode=mode,
        buffering=buffering,
        encoding=encoding,
        errors=errors,
        newline=newline,
        compression=compression,
        compresslevel=compresslevel,
        format=format,
        check=check,
        preset=preset,
        filters=filters,
    ) as output:
        if delimiter:
            output.write(iterable.make_string(delimiter))
        else:
            output.write(str(iterable))


def to_jsonl(
    iterable,
    path,
    mode: Union[Literal["r", "w", "x", "a", "b", "t", "+"], str] = "wb",
    compression: Union[None, Literal["gz", "gzip", "lzma", "xz", "bz2"]] = None,
    ensure_ascii=False,
):
    """
    Saves the sequence to a jsonl file. Each element is mapped using json.dumps then written
    with a newline separating each element.
    :param path: path to write file
    :param mode: mode to write in, defaults to 'w' to overwrite contents
    :param compression: compression format
    :param ensure_ascii: ensure_ascii
    """
    dumps = partial(json.dumps, ensure_ascii=ensure_ascii)
    with universal_write_open(path, mode=mode, compression=compression) as output:
        output.write((iterable.map(dumps).make_string("\n") + "\n").encode("utf-8"))


def to_json(
    iterable,
    path,
    root_array=True,
    mode: Union[Literal["r", "w", "x", "a", "b", "t", "+"], str] = "wt",
    compression: Union[None, Literal["gz", "gzip", "lzma", "xz", "bz2"]] = None,
    ensure_ascii=False,
):
    """
    Saves the sequence to a json file. If root_array is True, then the sequence will be written
    to json with an array at the root. If it is False, then the sequence will be converted from
    a sequence of (Key, Value) pairs to a dictionary so that the json root is a dictionary.
    :param path: path to write file
    :param root_array: write json root as an array or dictionary
    :param mode: file open mode
    """
    with universal_write_open(path, mode=mode, compression=compression) as output:
        if root_array:
            json.dump(list(iterable), output, ensure_ascii=ensure_ascii)
        else:
            json.dump(dict(iterable), output, ensure_ascii=ensure_ascii)


def to_csv(
    iterable,
    path,
    mode: Union[Literal["r", "w", "x", "a", "b", "t", "+"], str] = "wt",
    dialect="excel",
    compression: Union[None, Literal["gz", "gzip", "lzma", "xz", "bz2"]] = None,
    newline="",
    **fmtparams
):
    """
    Saves the sequence to a csv file. Each element should be an iterable which will be expanded
    to the elements of each row.
    :param path: path to write file
    :param mode: file open mode
    :param dialect: passed to csv.writer
    :param fmtparams: passed to csv.writer
    """

    if "b" in mode:
        newline = None

    with universal_write_open(
        path, mode=mode, compression=compression, newline=newline
    ) as output:
        csv_writer = csv.writer(output, dialect=dialect, **fmtparams)
        for row in iterable:
            csv_writer.writerow([str(element) for element in row])
