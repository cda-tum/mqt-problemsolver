[![CodeCov](https://github.com/cda-tum/MQTProblemSolver/actions/workflows/coverage.yml/badge.svg)](https://github.com/cda-tum/MQTProblemSolver/actions/workflows/coverage.yml)
[![Deploy to PyPI](https://github.com/cda-tum/MQTProblemSolver/actions/workflows/deploy.yml/badge.svg)](https://github.com/cda-tum/MQTProblemSolver/actions/workflows/deploy.yml)

<p align="center">
  <a href="https://mqt.readthedocs.io">
   <picture>
     <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/cda-tum/mqt/main/docs/_static/mqt_light.png" width="60%">
     <img src="https://raw.githubusercontent.com/cda-tum/mqt/main/docs/_static/mqt_dark.png" width="60%">
   </picture>
  </a>
</p>

# MQT ProblemSolver

This repository covers the implementations of multiple research papers in the domain of quantum computing:

1. [Towards an Automated Framework for Realizing Quantum Computing Solutions](#towards-an-automated-framework-for-realizing-quantum-computing-solutions)
2. [A Hybrid Classical Quantum Computing Approach to the Satellite Mission Planning Problem](#a-hybrid-classical-quantum-computing-approach-to-the-satellite-mission-planning-problem)
3. [Reducing the Compilation Time of Quantum Circuits Using Pre-Compilation on the Gate Level](#reducing-the-compilation-time-of-quantum-circuits-using-pre-compilation-on-the-gate-level)
4. [Utilizing Resource Estimation for the Development of Quantum Computing Applications](#utilizing-resource-estimation-for-the-development-of-quantum-computing-applications)
5. [Towards Equivalence Checking of Classical Circuits Using Quantum Computing](#towards-equivalence-checking-of-classical-circuits-using-quantum-computing)

In the following, each implementation is briefly introduced.

# Towards an Automated Framework for Realizing Quantum Computing Solutions

MQT ProblemSolver provides a framework to utilize quantum computing as a technology for users with little to no
quantum computing knowledge that is part of the Munich Quantum Toolkit (MQT) developed by the [Chair for Design Automation](https://www.cda.cit.tum.de/) at the [Technical University of Munich](https://www.tum.de/).
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

# A Hybrid Classical Quantum Computing Approach to the Satellite Mission Planning Problem

Additional to the two case studies, we provide a more complex example for the satellite mission planning problem.
The goal is to maximize the accumulated values of all images taken by the satellite while it is often not possible
to take all images since the satellite must rotate and adjust its optics.

In the following example, there are five to-be-captured locations which their assigned value.

<p align="center">
<img src="img/satellite_mission_planning_problem.png" height=200px>
</p>

# Reducing the Compilation Time of Quantum Circuits Using Pre-Compilation on the Gate Level

Every quantum computing application must be encoded into a quantum circuit and then compiled for a specific device.
This lengthy compilation process is a key bottleneck and intensifies for recurring problems---each of which requires
a new compilation run thus far.

<p align="center">
<img src="img/workflow_old.png">
</p>

Pre-compilation is a promising approach to overcome this bottleneck.
Beginning with a problem class and suitable quantum algorithm, a **predictive encoding** scheme is applied to encode a
representative problem instance into a general-purpose quantum circuit for that problem class.
Once the real problem instance is known, the previously constructed circuit only needs to be
**adjusted**—with (nearly) no compilation necessary:

<p align="center">
<img src="img/workflow_new.png">
</p>
Following this approach, we provide a pre-compilation module that can be used to precompile QAOA circuits
for the MaxCut problem.

# Utilizing Resource Estimation for the Development of Quantum Computing Applications

Resource estimation is a promising alternative to actually execute quantum circuits on real quantum hardware which is
currently restricted by the number of qubits and the error rates. By estimating the resources needed for a quantum circuit,
the development of quantum computing applications can be accelerated without the need to wait for the availability of
large-enough quantum hardware.

In `resource_estimation/RE_experiments.py`, we evaluate the resources to calculate the ground state energy of a
Hamiltonian to chemical accuracy of 1 mHartree using the qubitization quantum simulation algorithm. The Hamiltonian
describes the 64 electron and 56 orbital active space of one of the stable intermediates in the ruthenium-catalyzed
carbon fixation cycle

In this evaluation, we investigate

- different qubit technologies,
- the impact of the maximal number of T factories,
- different design trade-offs, and
- hypothesis on how quantum hardware might improve and how it affects the required resources.

# Towards Equivalence Checking of Classical Circuits Using Quantum Computing

Equivalence checking, i.e., verifying whether two circuits realize the same functionality or not, is a typical task in the semiconductor industry. Due to the fact, that the designs grow faster than the ability to efficiently verify them, all alternative directions to close the resulting verification gap should be considered. In the `equivalence_checker.py` module, our approach to this problem by utilizing quantum computing is implemented in two versions:

- With `try_parameter_combinations()` different parameter combinations can be evaluated with miters for which the counter examples are known
- `find_counter_examples()` is used to find counter examples for a miter for which counter examples should be found in the case of non-equivalence

# Usage

MQT ProblemSolver is available via [PyPI](https://pypi.org/project/mqt.problemsolver/):

```console
(venv) $ pip install mqt.problemsolver
```

# References

In case you are using MQT ProblemSolver in your work, we would be thankful if you referred to it by citing the following publication:

```bibtex
@INPROCEEDINGS{quetschlich2023mqtproblemsolver,
    title     = {{Towards an Automated Framework for Realizing Quantum Computing Solutions}},
    author    = {N. Quetschlich and L. Burgholzer and R. Wille},
    booktitle = {International Symposium on Multiple-Valued Logic (ISMVL)},
    year      = {2023},
}
```

which is also available on arXiv:
[![a](https://img.shields.io/static/v1?label=arXiv&message=2210.14928&color=inactive&style=flat-square)](https://arxiv.org/abs/2210.14928)

In case you are using our Satellite Mission Planning Problem approach, we would be thankful if you referred to it by citing the following publication:

```bibtex
@INPROCEEDINGS{quetschlich2023satellite,
    title     = {{A Hybrid Classical Quantum Computing Approach to the Satellite Mission Planning Problem}},
    author    = {N. Quetschlich and V. Koch and L. Burgholzer and R. Wille},
    booktitle = {IEEE International Conference on Quantum Computing and Engineering (QCE)},
    year      = {2023},
}
```

which is also available on arXiv:
[![a](https://img.shields.io/static/v1?label=arXiv&message=2308.00029&color=inactive&style=flat-square)](https://arxiv.org/abs/2308.00029)

In case you are using our Pre-Compilation approach, we would be thankful if you referred to it by citing the following publication:

```bibtex
@INPROCEEDINGS{quetschlich2023precompilation,
    title     = {{Reducing the Compilation Time of Quantum Circuits Using Pre-Compilation on the Gate Level}},
    author    = {N. Quetschlich and L. Burgholzer and R. Wille},
    booktitle = {IEEE International Conference on Quantum Computing and Engineering (QCE)},
    year      = {2023},
}
```

which is also available on arXiv:
[![a](https://img.shields.io/static/v1?label=arXiv&message=2305.04941&color=inactive&style=flat-square)](https://arxiv.org/abs/2305.04941)

In case you are using our Resources Estimation approach, we would be thankful if you referred to it by citing the following publication:

```bibtex
@misc{quetschlich2024resource_estimation,
    title           = {{Utilizing Resource Estimation for the Development of Quantum Computing Applications}},
    author          = {N. Quetschlich and M. Soeken and P. Murali and R. Wille},
    year            = {2024},
    eprint          = {2402.12434},
    archivePrefix   = {arXiv},
    booktitle       = {IEEE International Conference on Quantum Computing and Engineering (QCE)},
}
```

In case you are using our Equivalence-Checking approach, we would be thankful if you referred to it by citing the following publication:

```bibtex
@INPROCEEDINGS{quetschlich2024equivalence_checking,
    title     = {{Towards Equivalence Checking of Classical Circuits Using Quantum Computing}},
    author    = {N. Quetschlich and T. Forster and A. Osterwind and D. Helms and R. Wille},
    booktitle = {IEEE International Conference on Quantum Computing and Engineering (QCE)},
    year      = {2024},
}
```

which is also available on arXiv:
[![a](https://img.shields.io/static/v1?label=arXiv&message=2402.12434&color=inactive&style=flat-square)](https://arxiv.org/abs/2402.12434)

## Acknowledgements

This project received funding from the European Research Council (ERC) under the European Union's Horizon 2020 research
and innovation program (DA QC, grant agreement No. 101001318 and MILLENION, grant agreement No. 101114305), was part
of the Munich Quantum Valley, which is supported by the Bavarian state government with funds from the Hightech Agenda
Bayern Plus, and has been supported by the BMWK on the basis of a decision by the German Bundestag through project
QuaST, as well as by the BMK, BMDW, the State of Upper Austria in the frame of the COMET program, and the QuantumReady
project within Quantum Austria (managed by the FFG).

<p align="center">
<picture>
<source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/cda-tum/mqt/main/docs/_static/tum_dark.svg" width="28%">
<img src="https://raw.githubusercontent.com/cda-tum/mqt/main/docs/_static/tum_light.svg" width="28%" alt="TUM Logo">
</picture>
<picture>
<img src="https://raw.githubusercontent.com/cda-tum/mqt/main/docs/_static/logo-bavaria.svg" width="16%" alt="Coat of Arms of Bavaria">
</picture>
<picture>
<source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/cda-tum/mqt/main/docs/_static/erc_dark.svg" width="24%">
<img src="https://raw.githubusercontent.com/cda-tum/mqt/main/docs/_static/erc_light.svg" width="24%" alt="ERC Logo">
</picture>
<picture>
<img src="https://raw.githubusercontent.com/cda-tum/mqt/main/docs/_static/logo-mqv.svg" width="28%" alt="MQV Logo">
</picture>
</p>
