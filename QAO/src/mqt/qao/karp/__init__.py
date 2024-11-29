"""Karp Module."""

from __future__ import annotations

import logging

from .karp_graphs import KarpGraphs
from .karp_number import KarpNumber
from .karp_sets import KarpSets

__all__ = ["KarpGraphs", "KarpNumber", "KarpSets"]

logger = logging.getLogger("mqt-qao-karp")

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
logger_formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(logger_formatter)

logger.addHandler(console_handler)
logger.setLevel(logging.DEBUG)
