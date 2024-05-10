from __future__ import annotations

import logging

from src.mqt.qao.constraints import Constraints
from src.mqt.qao.objectivefunction import ObjectiveFunction
from src.mqt.qao.problem import Problem
from src.mqt.qao.solvers import Solver
from src.mqt.qao.variables import Variables

__all__ = [
    "Variables",
    "Constraints",
    "ObjectiveFunction",
    "Problem",
    "Solver",
]

logger = logging.getLogger("mqt-qao")

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
logger_formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(logger_formatter)
logger.addHandler(console_handler)
logger.setLevel(logging.DEBUG)
