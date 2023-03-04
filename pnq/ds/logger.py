import logging
from logging import Logger
from typing import Dict

from pnq import query


def get_loggers():
    loggers: Dict[str, Logger] = logging.root.manager.loggerDict  # type: ignore
    return query(loggers)
