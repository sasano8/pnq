import os
from pathlib import Path

import unasync

# additional_replacements = {
#     # We want to rewrite to 'Transport' instead of 'SyncTransport', etc
#     "AsyncTransport": "Transport",
#     "AsyncElasticsearch": "Elasticsearch",
#     # We don't want to rewrite this class
#     "AsyncSearchClient": "AsyncSearchClient",
# }

PACKAGE = "pnq"


CODE_ROOT = Path(__file__).absolute().parent

rules = [
    unasync.Rule(
        fromdir=str(CODE_ROOT / "_async"),
        todir=str(CODE_ROOT / "_sync_generate"),
        # additional_replacements={"Listable": "map_iter", "Listable": "map_iter"},
    ),
]

filepaths = []
for root, _, filenames in os.walk(CODE_ROOT / "_async"):
    for filename in filenames:
        if filename.rpartition(".")[-1] in (
            "py",
            "pyi",
        ) and not filename.startswith("utils.py"):
            filepaths.append(os.path.join(root, filename))

# create _sync from _async
unasync.unasync_files(filepaths, rules)


# TODO: このモジュールはパッケージの外に出したい
