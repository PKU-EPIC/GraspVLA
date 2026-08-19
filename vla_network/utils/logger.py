"""Small project-local logger used by training and serving."""

from __future__ import annotations

import logging
import time


class Log:
    """Compatibility wrapper for the logging calls used in this project."""

    def __init__(self) -> None:
        self.silent = False
        self.init_time = time.perf_counter()
        self._logger = logging.getLogger("vla_network")
        if not self._logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(message)s"))
            self._logger.addHandler(handler)
        self._logger.setLevel(logging.INFO)
        self._logger.propagate = False

    def set_silent(self, silent: bool = True) -> None:
        self.silent = silent

    def _write(self, level: int, message: str) -> None:
        if not self.silent:
            elapsed = time.perf_counter() - self.init_time
            self._logger.log(level, f"{elapsed:.3f}s {message}")

    def debug(self, message: str) -> None:
        self._write(logging.DEBUG, message)

    def info(self, message: str) -> None:
        self._write(logging.INFO, message)

    def warn(self, message: str) -> None:
        self._write(logging.WARNING, message)

    warning = warn

    def error(self, message: str) -> None:
        self._write(logging.ERROR, message)

    def warn_when(self, condition: bool, message: str) -> None:
        if condition:
            self.warn(message)

    def error_when(self, condition: bool, message: str) -> None:
        if condition:
            self.error(message)


log = Log()
