from __future__ import annotations

import logging

from mqt.qao.constraints import Constraints
from mqt.qao.objectivefunction import ObjectiveFunction
from mqt.qao.problem import Problem
from mqt.qao.solvers import Solution, Solver
from mqt.qao.variables import Variable, Variables

__all__ = [
    "Constraints",
    "ObjectiveFunction",
    "Problem",
    "Solution",
    "Solver",
    "Variable",
    "Variables",
]

logger = logging.getLogger("mqt-qao")

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
logger_formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(logger_formatter)
logger.addHandler(console_handler)
logger.setLevel(logging.DEBUG)
