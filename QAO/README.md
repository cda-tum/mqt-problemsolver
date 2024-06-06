[![PyPI](https://img.shields.io/pypi/v/mqt.qao?logo=pypi&style=flat-square)](https://pypi.org/project/mqt.qao/)
![OS](https://img.shields.io/badge/os-linux%20%7C%20macos%20%7C%20windows-blue?style=flat-square)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg?style=flat-square)](https://opensource.org/licenses/MIT)
[![CI](https://img.shields.io/github/actions/workflow/status/cda-tum/mqt-qao/ci.yml?branch=main&style=flat-square&logo=github&label=ci)](https://github.com/cda-tum/mqt-qao/actions/workflows/ci.yml)
[![CD](https://img.shields.io/github/actions/workflow/status/cda-tum/mqt-qao/cd.yml?style=flat-square&logo=github&label=cd)](https://github.com/cda-tum/mqt-qao/actions/workflows/cd.yml)
[![Documentation](https://img.shields.io/readthedocs/qao?logo=readthedocs&style=flat-square)](https://mqt.readthedocs.io/projects/qao)
[![codecov](https://img.shields.io/codecov/c/github/cda-tum/mqt-qao?style=flat-square&logo=codecov)](https://codecov.io/gh/cda-tum/mqt-qao)

<p align="center">
  <a href="https://mqt.readthedocs.io">
   <picture>
     <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/cda-tum/mqt/main/docs/_static/mqt_light.png" width="60%">
     <img src="https://raw.githubusercontent.com/cda-tum/mqt/main/docs/_static/mqt_dark.png" width="60%">
   </picture>
  </a>
</p>

# MQT Quantum Auto Optimizer: Automatic Framework for Solving Optimization Problems with Quantum Computers

MQT Quantum Auto Optimizer is a framework that allows one to automatically translate an optimization problem into a quantum-compliant formulation and to solve it with one of the main quantum solvers (Quantum Annealer, Quantum Approximate Optimization Algorithm, Variational Quantum Eigensolver and Grover Adaptive Search)

MQT Quantum Auto Optimizer is part of the [Munich Quantum Toolkit (MQT)](https://mqt.readthedocs.io/) developed by the [Chair for Design Automation](https://www.cda.cit.tum.de/) at the [Technical University of Munich](https://www.tum.de/). This framework has been developed in collaboration with the [VLSI Lab](https://www.vlsilab.polito.it/) of [Politecnico di Torino](https://www.polito.it).

If you have any questions, feel free to create a [discussion](https://github.com/cda-tum/mqt-qao/discussions) or an [issue](https://github.com/cda-tum/mqt-qao/issues) on [GitHub](https://github.com/cda-tum/mqt-qao).

## Getting Started

`mqt-qao` is available via [PyPI](https://pypi.org/project/mqt.qao/).

```console
(venv) $ pip install mqt.qao
```

The following code gives an example on the usage:

```python3
from mqt.qao import Constraints, ObjectiveFunction, Problem, Solver, Variables

# Declaration of the problem variables
var = Variables()
a = var.add_binary_variable("a")
b = var.add_discrete_variable("b", [-1, 1, 3])
c = var.add_continuous_variable("c", -2, 2, 0.25)

# declaration of the objective functions involved in the problem
obj_func = ObjectiveFunction()
obj_func.add_objective_function(a + b * c + c**2)

# Declaration of the constraints
cst = Constraints()
cst.add_constraint("b + c >= 2", variable_precision=True)

# Creation of the problem
prb = Problem()
prb.create_problem(var, cst, obj_func)

# Solve the problem with the Dwave Quantum Annealer
solution = Solver().solve_Dwave_quantum_annealer(prb, token=token)
```

**Detailed documentation and examples are available at [ReadTheDocs](https://mqt.readthedocs.io/projects/qao).**

## Acknowledgements

The Munich Quantum Toolkit has been supported by the European
Research Council (ERC) under the European Union's Horizon 2020 research and innovation program (grant agreement
No. 101001318), the Bavarian State Ministry for Science and Arts through the Distinguished Professorship Program, as well as the
Munich Quantum Valley, which is supported by the Bavarian state government with funds from the Hightech Agenda Bayern Plus.

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
