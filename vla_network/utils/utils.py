import os
from os.path import abspath, dirname

VLA_NETWORK_ROOT = dirname(dirname(abspath(__file__)))


def is_main_rank() -> bool:
    return int(os.environ.get("RANK", "0")) == 0
