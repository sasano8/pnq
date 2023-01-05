import logging
from logging import Logger
from pnq import query
from typing import Dict


def get_loggers():
    loggers: Dict[str, Logger] = logging.root.manager.loggerDict  # type: ignore
    return query(loggers)
