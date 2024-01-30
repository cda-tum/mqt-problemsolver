"""A package for generating QUBO formulations automatically from a set of constraints QUBOs for different problem classes.

It allows users to create a `QUBOGenerator` object, and graudally add penalty terms and constraints to it.
When done, the object can be used to construct a QUBO formulation of the project on multiple granularity levels.

## Available Subpackages
- `pathfinder`: This module implements the pathfinding functionalities of the QUBOMaker.
"""

from __future__ import annotations

from . import pathfinder
from .graph import Graph
from .qubo_generator import QUBOGenerator
from .tsplib import get_qubo_generator
from .utils import optimise_classically, print_matrix

__all__ = [
    "pathfinder",
    "Graph",
    "QUBOGenerator",
    "get_qubo_generator",
    "optimise_classically",
    "print_matrix",
]
