import csv
import json
from functools import partial
from typing import Iterable, Mapping, Union

from typing_extensions import Literal

from ..io import universal_write_open
from .finalizers import first_or


async def to_file(
    source,
    path,
    delimiter: str = None,
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
        if delimiter is None:
            bracket_start = "["
            bracket_end = "]"
            delimiter = ","
        else:
            bracket_start = ""
            bracket_end = ""
            delimiter = delimiter

        undefined = object()
        it = source.__aiter__()
        first = await first_or(it, undefined)

        if first is undefined:
            if bracket_start:
                output.write(bracket_start)
                output.write(bracket_end)
            return

        if bracket_start:
            output.write(bracket_start)

        output.write(str(first))

        async for row in it:
            output.write(delimiter + str(row))

        if bracket_end:
            output.write(bracket_end)


async def to_json(
    source,
    path,
    as_dict=False,
    mode: Union[Literal["r", "w", "x", "a", "b", "t", "+"], str] = "wt",
    compression: Union[None, Literal["gz", "gzip", "lzma", "xz", "bz2"]] = None,
    ensure_ascii=False,
    indent=2,
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
        dumps = partial(json.dumps, ensure_ascii=ensure_ascii, indent=indent)
        s_indent = " " * indent

        if not as_dict:
            output.write("[")
        else:
            output.write("{")

        undefined = object()
        it = source.__aiter__()
        first_ = await first_or(it, undefined)

        if first_ is not undefined:
            if not as_dict:
                output.write("\n" + s_indent + dumps(first_))
                s_indent = ",\n" + s_indent

                async for x in it:
                    output.write(s_indent + dumps(x))
            else:
                k, v = first_
                output.write("\n" + s_indent + str(dumps(k)) + ": " + dumps(v))
                s_indent = ",\n" + s_indent

                async for k, v in it:
                    output.write(s_indent + str(dumps(k)) + ": " + dumps(v))

        if not as_dict:
            output.write("\n]\n")
        else:
            output.write("\n}\n")


async def to_jsonl(
    source,
    path,
    mode: Union[Literal["r", "w", "x", "a", "b", "t", "+"], str] = "wt",
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
        async for x in source:
            output.write((dumps(x) + "\n"))


async def to_csv(
    source,
    path,
    mode: Union[Literal["r", "w", "x", "a", "b", "t", "+"], str] = "wt",
    header: Union[None, bool, Iterable[str]] = True,
    # columns: Iterable[str] = None,
    skip_first: bool = False,
    delimiter: str = ",",
    dialect="excel",
    compression: Union[None, Literal["gz", "gzip", "lzma", "xz", "bz2"]] = None,
    newline="",
    **fmtparams,
):
    """
    Saves the sequence to a csv file. Each element should be an iterable which will be expanded
    to the elements of each row.
    :param path: path to write file
    :param mode: file open mode
    :param dialect: passed to csv.writer
    :param fmtparams: passed to csv.writer
    """

    if "fieldnames" in fmtparams:
        raise ValueError("fieldnames is not supported. use header.")

    if "b" in mode:
        newline = None

    with universal_write_open(
        path, mode=mode, compression=compression, newline=newline
    ) as output:
        # csv_writer = csv.writer(output, dialect=dialect, **fmtparams)

        if header is None or isinstance(header, int):
            output_header = bool(header)
            columns = None
        else:
            output_header = True
            columns = header

        header_printed = False

        if output_header and skip_first and columns is None:
            raise ValueError(
                f"The behavior is undefined. header:{header} skip_first:{skip_first}"
            )

        if output_header and columns is not None:
            csv.writer(
                output, dialect=dialect, delimiter=delimiter, **fmtparams
            ).writerow(columns)
            header_printed = True

        undefined = object()
        it = source.__aiter__()
        first_ = await first_or(it, undefined)

        if first_ is undefined:
            return

        if isinstance(first_, Mapping):
            if columns is None:
                columns = list(first_.keys())
            else:
                ...

            writer = csv.DictWriter(
                output,
                fieldnames=columns,
                dialect=dialect,
                delimiter=delimiter,
                **fmtparams,
            )
            if output_header and not header_printed:
                writer.writeheader()

            if not skip_first:
                writer.writerow(first_)

        else:

            if columns is None:
                columns = first_
            else:
                ...

            writer = csv.writer(
                output,
                dialect=dialect,
                delimiter=delimiter,
                **fmtparams,
            )
            if not skip_first:
                writer.writerow(first_)

        async for row in it:
            writer.writerow(row)
