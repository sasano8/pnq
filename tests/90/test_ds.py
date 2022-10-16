import asyncio
import os
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List

import pytest

from pnq.ds import docker, filesystem, schedule, types


@pytest.fixture
def tmpdirs():
    cd = os.getcwd()
    with TemporaryDirectory() as d1, TemporaryDirectory() as d2:
        d1 = Path(d1).absolute()
        d2 = Path(d2).absolute()
        yield d1, d2
    os.chdir(cd)


def test_filesystem(tmpdirs: List[Path]):
    tmp1, tmp2 = tmpdirs

    f1 = tmp1 / "f1"
    d1 = tmp1 / "d1"
    d2 = tmp1 / "d2"

    os.chdir(tmp1)

    if "is empty current directory":
        assert filesystem.ls().map(Path.absolute).to(set) == set()
        assert filesystem.files().map(Path.absolute).to(set) == set()
        assert filesystem.dirs().map(Path.absolute).to(set) == set()

    if "query file":
        p = f1
        p.touch(exist_ok=False)
        assert filesystem.ls().map(Path.absolute).to(set) == set([f1])
        assert filesystem.files().map(Path.absolute).to(set) == set([f1])
        assert filesystem.dirs().map(Path.absolute).to(set) == set([])

    if "query directory":
        p = d1
        p.mkdir(exist_ok=False)
        assert filesystem.ls().map(Path.absolute).to(set) == set([f1, d1])
        assert filesystem.files().map(Path.absolute).to(set) == set([f1])
        assert filesystem.dirs().map(Path.absolute).to(set) == set([d1])

    if "query directory2":
        p = d2
        p.mkdir(exist_ok=False)
        assert filesystem.ls().map(Path.absolute).to(set) == set([f1, d1, d2])
        assert filesystem.files().map(Path.absolute).to(set) == set([f1])
        assert filesystem.dirs().map(Path.absolute).to(set) == set([d1, d2])

    os.chdir(tmp2)

    f10 = tmp2 / "f110"
    d10 = tmp2 / "d10"
    d20 = d10 / "d20"
    f20 = d10 / "f20"
    f30 = d10 / "f30"

    if "check default args":
        # fmt: off
        assert filesystem.ls(pathname="*", root_dir=None, dir_fd=None, recursive=False).map(Path.absolute).to(set) == set()  # noqa
        assert filesystem.files(pathname="*", root_dir=None, dir_fd=None, recursive=False).map(Path.absolute).to(set) == set()  # noqa
        assert filesystem.dirs(pathname="*", root_dir=None, dir_fd=None, recursive=False).map(Path.absolute).to(set) == set()  # noqa
        # fmt: on

    if "check recursive":
        f10.touch(exist_ok=False)
        d10.mkdir(exist_ok=False)
        d20.mkdir(exist_ok=False)
        f20.touch(exist_ok=False)
        f30.touch(exist_ok=False)

        # fmt: off
        assert filesystem.ls(pathname="**", recursive=False).map(Path.absolute).to(set) == set([f10, d10])  # noqa
        assert filesystem.files(pathname="**", recursive=False).map(Path.absolute).to(set) == set([f10])  # noqa
        assert filesystem.dirs(pathname="**", recursive=False).map(Path.absolute).to(set) == set([d10])  # noqa
        # fmt: on

        # fmt: off
        assert filesystem.ls(pathname="**", recursive=True).map(Path.absolute).to(set) == set([f10, d10, d20, f20, f30])  # noqa
        assert filesystem.files(pathname="**", recursive=True).map(Path.absolute).to(set) == set([f10, f20, f30])  # noqa
        assert filesystem.dirs(pathname="**", recursive=True).map(Path.absolute).to(set) == set([d10, d20])  # noqa
        # fmt: on

    if "check pathlib":
        p = Path(tmp2)
        # fmt: off
        assert filesystem.ls(pathname=p / "**", recursive=False).map(Path.absolute).to(set) == set([f10, d10])  # noqa
        # fmt: on


def test_schedule_tick():
    async def main(is_async):
        if is_async:
            times = await schedule.tick_async(seconds=1).take(3)
            assert len(await schedule.tick_async(seconds=1).take(1)) == 1  # 再実行できるか確認
        else:
            times = schedule.tick(seconds=1).take(3).result()
            assert len(schedule.tick(seconds=1).take(1).result()) == 1  # 再実行できるか確認

        assert len(times) == 3
        assert all(isinstance(x, datetime) for x in times)

        t1 = (times[1] - times[0]).total_seconds()
        t2 = (times[2] - times[1]).total_seconds()

        # おおよそ１秒ならよしとする
        assert 1 < t1 and t1 < 1.005
        assert 1 < t2 and t2 < 1.005

    asyncio.run(main(False))
    asyncio.run(main(True))


def test_schedule_weekday():
    assert schedule.weekdays.items().to(dict) == {
        0: "Monday",
        1: "Tuesday",
        2: "Wednesday",
        3: "Thursday",
        4: "Friday",
        5: "Saturday",
        6: "Sunday",
    }

    assert schedule.isoweekdays.items().to(dict) == {
        1: "Monday",
        2: "Tuesday",
        3: "Wednesday",
        4: "Thursday",
        5: "Friday",
        6: "Saturday",
        7: "Sunday",
    }


def test_types():
    import platform

    # TODO: 3.10以降では削除
    V_3_10 = int(platform.python_version_tuple()[1]) >= 10
    V_3_09 = int(platform.python_version_tuple()[1]) >= 9

    res = types.items().to(dict)
    if V_3_10:
        assert len(res) == 73
    if V_3_09:
        assert len(res) == 67
    else:
        assert len(res) == 61

    assert isinstance(res, dict)


def test_docker():
    # TODO: 例外処理を書く
    client = docker.Client()
    commands = client.commands().to(list)
    assert commands == [
        "images",
        "containers",
        "networks",
        "volumes",
        "nodes",
        "services",
        "tasks",
        "secrets",
        "configs",
    ]
    for command in commands:
        query = getattr(client, command)
        assert len(query().to(list)) >= 0
