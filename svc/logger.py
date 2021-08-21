import logging
import sys
from pythonjsonlogger import jsonlogger


def init_logger():
    log = logging.getLogger()
    log.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(jsonlogger.JsonFormatter(timestamp=True))
    log.addHandler(handler)
