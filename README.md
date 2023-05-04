[![CodeCov](https://github.com/cda-tum/MQTProblemSolver/actions/workflows/coverage.yml/badge.svg)](https://github.com/cda-tum/MQTProblemSolver/actions/workflows/coverage.yml)
[![Deploy to PyPI](https://github.com/cda-tum/MQTProblemSolver/actions/workflows/deploy.yml/badge.svg)](https://github.com/cda-tum/MQTProblemSolver/actions/workflows/deploy.yml)

<p align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/cda-tum/MQTProblemSolver/main/img/mqt_light.png" width="60%">
  <img src="https://raw.githubusercontent.com/cda-tum/MQTProblemSolver/main/img/mqt_dark.png" width="60%">
</picture>
</p>

# MQT ProblemSolver

MQT ProblemSolver is a framework to utilize quantum computing as a technology for users with little to no
quantum computing knowledge.
All necessary quantum parts are embedded by domain experts while the interfaces provided are similar to the ones
classical solver provide:

<p align="center">
<img src="img/framework.png" height=300px>
</p>

When provided with a problem description, MQT ProblemSolver offers a selection of implemented quantum algorithms.
The user just has to chose one and all further (quantum) calculation steps are encapsulated within MQT ProblemSolver.
After the calculation finished, the respective solution is returned - again in the same format as classical
solvers use.

In the current implementation, two case studies are conducted:

1. A SAT Problem: Constraint Satisfaction Problem
2. A Graph-based Optimization Problem: Travelling Salesman Problem

## A SAT Problem: Constraint Satisfaction Problem

This exemplary implementation can be found in the [CSP_example.ipynb](notebooks/csp_example.ipynb) Jupyter notebook.
Here, the solution to a Kakuro riddle with a 2x2 grid can be solved for arbitrary sums `s0` to `s3`:

<p align="center">
<img src="img/kakuro.png" height=100px>
</p>

MQT ProblemSolver will return valid values to `a`, `b`, `c`, and `d` if a solution exists.

## A Graph-based Optimization Problem: Travelling Salesman Problem

This exemplary implementation can be found in the [TSP_example.ipynb](notebooks/tsp_example.ipynb) Jupyter notebook.
Here, the solution to a Travelling Salesman Problem with 4 cities can be solved for arbitrary distances `dist_1_2` to `dist_3_4`between the cities.

<p align="center">
<img src="img/tsp.png" height=200px>
</p>

MQT ProblemSolver will return the shortest path visiting all cities as a list.

## Satellite Mission Planning Problem

Additional to the two case studies, we provide a more complex example for the satellite mission planning problem.
The goal is to maximize the accumulated values of all images taken by the satellite while it is often not possible
to take all images since the satellite must rotate and adjust its optics.

In the following example, there are five to-be-captured locations which their assigned value.

<p align="center">
<img src="img/satellite_mission_planning_problem.png" height=200px>
</p>

# Pre-Compilation

Every quantum computing application must be encoded into a quantum circuit and then compiled for a specific device.
This lengthy compilation process is a key bottleneck and intensifies for recurring problems---each of which requires
a new compilation run thus far.

Pre-compilation is a promising approach to overcome this bottleneck.
Beginning with a problem class and suitable quantum algorithm, a **predictive encoding** scheme is applied to encode a
representative problem instance into a general-purpose quantum circuit for that problem class.
Once the real problem instance is known, the previously constructed circuit only needs to be
**adjusted**—with (nearly) no compilation necessary.

Following this approach, we provide a pre-compilation module that can be used to precompile QAOA circuits
for the MaxCut problem.

# Usage

MQT ProblemSolver is available via [PyPI](https://pypi.org/project/mqt.problemsolver/):

```console
(venv) $ pip install mqt.problemsolver
```

# Repository Structure

```
.
├── src
│ └── mqt
│     └── problemsolver
│     │  └── csp.py
│     │  └── tsp.py
│     └── satelitesolver
│     │  └── ...
│     └── precompilation
│        └── ...
└── notebooks
│     └── problemsolver
│        └── csp_example.ipynb
│        └── tsp_example.ipynb
│        └── satellitesolver
│           └── satellitesolver_example.ipynb
```

# References

In case you are using MQT ProblemSolver in your work, we would be thankful if you referred to it by citing the following publication:

```bibtex
@INPROCEEDINGS{quetschlich2023mqtproblemsolver,
    author    = {N. Quetschlich and L. Burgholzer and R. Wille},
    title     = {{Towards an Automated Framework for Realizing Quantum Computing Solutions}},
    booktitle = {International Symposium on Multiple-Valued Logic (ISMVL)},
    year      = {2023},
}
```
