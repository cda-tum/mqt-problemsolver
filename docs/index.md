---
file_format: mystnb
kernelspec:
  name: python3
mystnb:
  number_source_lines: true
---

# MQT ProblemSolver

MQT ProblemSolver provides a framework to utilize quantum computing as a technology for users with little to no quantum computing knowledge
It is developed as part of the _{doc}`Munich Quantum Toolkit (MQT) <mqt:index>`_.

This documentation covers the implementations of multiple research papers in the domain of quantum computing.

## Towards an Automated Framework for Realizing Quantum Computing Solutions

All necessary quantum parts are embedded by domain experts while the interfaces provided are similar to the ones classical solver provide:

<p align="center">
<img src="_static/framework.png" height=300px>
</p>

When provided with a problem description, MQT ProblemSolver offers a selection of implemented quantum algorithms.
The user just has to choose one and all further (quantum) calculation steps are encapsulated within MQT ProblemSolver.
After the calculation finished, the respective solution is returned—again in the same format as classical solvers use.

For more details, see {cite:p}`quetschlich2023mqtproblemsolver`.

In the current implementation, two case studies are conducted:

1. A SAT Problem: Constraint Satisfaction Problem
2. A Graph-based Optimization Problem: Travelling Salesman Problem

### A SAT Problem: Constraint Satisfaction Problem

This exemplary implementation can be found in the [`csp_example.ipynb`](https://github.com/munich-quantum-toolkit/problemsolver/blob/main/notebooks/csp_example.ipynb) Jupyter notebook.
Here, the solution to a Kakuro riddle with a 2x2 grid can be solved for arbitrary sums `s0` to `s3`:

<p align="center">
<img src="_static/kakuro.png" height=100px>
</p>

MQT ProblemSolver will return valid values to `a`, `b`, `c`, and `d` if a solution exists.

### A Graph-based Optimization Problem: Travelling Salesman Problem

This exemplary implementation can be found in the [`tsp_example.ipynb`](https://github.com/munich-quantum-toolkit/problemsolver/blob/main/notebooks/tsp_example.ipynb) Jupyter notebook.
Here, the solution to a Travelling Salesman Problem with 4 cities can be solved for arbitrary distances `dist_1_2` to `dist_3_4`between the cities.

<p align="center">
<img src="_static/tsp.png" height=200px>
</p>

MQT ProblemSolver will return the shortest path visiting all cities as a list.

## A Hybrid Classical Quantum Computing Approach to the Satellite Mission Planning Problem

Additional to the two case studies, we provide a more complex example for the satellite mission planning problem.
The goal is to maximize the accumulated values of all images taken by the satellite while it is often not possible to take all images since the satellite must rotate and adjust its optics.

In the following example, there are five to-be-captured locations which their assigned value.

<p align="center">
<img src="_static/satellite_mission_planning_problem.png" height=200px>
</p>

For more details, see {cite:p}`quetschlich2023satellite`.

## Reducing the Compilation Time of Quantum Circuits Using Pre-Compilation on the Gate Level

Every quantum computing application must be encoded into a quantum circuit and then compiled for a specific device.
This lengthy compilation process is a key bottleneck and intensifies for recurring problems---each of which requires a new compilation run thus far.

<p align="center">
<img src="_static/workflow_old.png">
</p>

Pre-compilation is a promising approach to overcome this bottleneck.
Beginning with a problem class and suitable quantum algorithm, a **predictive encoding** scheme is applied to encode a representative problem instance into a general-purpose quantum circuit for that problem class.
Once the real problem instance is known, the previously constructed circuit only needs to be **adjusted**—with (nearly) no compilation necessary:

<p align="center">
<img src="_static/workflow_new.png">
</p>
Following this approach, we provide a pre-compilation module that can be used to precompile QAOA circuits for the MaxCut problem.

For more details, see {cite:p}`quetschlich2023precompilation`.

## Utilizing Resource Estimation for the Development of Quantum Computing Applications

Resource estimation is a promising alternative to actually execute quantum circuits on real quantum hardware which is currently restricted by the number of qubits and the error rates. By estimating the resources needed for a quantum circuit,
the development of quantum computing applications can be accelerated without the need to wait for the availability of large-enough quantum hardware.

In [`experiments.ipynb`](https://github.com/munich-quantum-toolkit/problemsolver/blob/main/notebooks/resource_estimation/experiments.ipynb), we evaluate the resources to calculate the ground state energy of a Hamiltonian to chemical accuracy of 1 mHartree using the qubitization quantum simulation algorithm.
The Hamiltonian describes the 64 electron and 56 orbital active space of one of the stable intermediates in the ruthenium-catalyzed carbon fixation cycle.

In this evaluation, we investigate

- different qubit technologies,
- the impact of the maximal number of T factories,
- different design trade-offs, and
- hypothesis on how quantum hardware might improve and how it affects the required resources.

For more details, see {cite:p}`quetschlich2024resource_estimation`.

## Towards Equivalence Checking of Classical Circuits Using Quantum Computing

Equivalence checking, i.e., verifying whether two circuits realize the same functionality or not, is a typical task in the semiconductor industry.
Due to the fact, that the designs grow faster than the ability to efficiently verify them, all alternative directions to close the resulting verification gap should be considered.
In this work, we consider the problem through the miter structure.
Here, two circuits to be checked are applied with the same primary inputs.
Then, for each pair of to-be-equal output bits, an exclusive-OR (XOR) gate is applied-evaluating to 1 if the two outputs generate different values (which only happens in the case of non-equivalence).
By OR-ing the outputs of all these XOR gates, eventually an indicator results that shows whether both circuits are equivalent.
Then, the goal is to determine an input assignment so that this indicator evaluates to 1 (providing a counter example that shows non-equivalence) or to prove that no such assignment exists (proving equivalence).

<p align="center">
<img src="_static/miter_structure.png" height=250px>
</p>

In the `equivalence_checking` module, our approach to this problem by utilizing quantum computing is implemented. There are two different ways to run this code.

- One to test, how well certain parameter combinations work.
  The parameters consist of the number of bits of the circuits to be verified, the threshold parameter delta (which is explained in detail in the paper), the fraction of input combinations that induce non-equivalence of the circuits (further called "counter examples"), the number of shots to run the quantum circuit for and the number of individual runs of the experiment.
  Multiple parameter combinations can be tested and exported as a `.csv`-file at a provided location.
- A second one to actually input a miter expression (in form of a string) together with some parameters independent from the miter (shots and delta) and use our approach to find the counter examples (if the circuits are non-equivalent).

These two implementations are provided by the functions `try_parameter_combinations()` and `find_counter_examples()`, respectively.
Examples for their usages are shown in the [`equivalence_checking_example.ipynb`](https://github.com/munich-quantum-toolkit/problemsolver/blob/main/notebooks/equivalence_checking/equivalence_checking_example.ipynb) Jupyter notebook.

For more details, see {cite:p}`quetschlich2024equivalence_checking`.

## Improving Hardware Requirements for Fault-Tolerant Quantum Computing by Optimizing Error Budget Distributions

Applying error correction to execute quantum circuits fault-tolerantly induces massive overheads in the required physical resources, often in the orders of magnitude.
This leads to thousands of qubits already for toy-sized quantum applications.
Obviously, these need to be reduced, for which the so-called error budget can be a particular lever.
Even though error correction is applied, a certain error rate still remains in the execution of the quantum circuit.
Hence, the end user defines a maximum tolerated error rate, the error budget, for the quantum application to be considered by the compiler.
Since an error-corrected quantum circuit consists of different parts, this error budget is distributed among these parts.
The way how it is distributed can have a significant effect on the resulting required resources.
To find an efficient distribution, we use resource estimation to evaluate different distributions as well as a machine learning model approach that automatically determines such efficient distributions for a given quantum circuit.

<p align="center">
<img src="_static/error_budget_approach.svg" height=150px>
</p>

For an examplary usage of the implementation, see below.

```{code-cell} ipython3
from mqt.problemsolver.resource_estimation.error_budget_optimization import (
    evaluate,
    generate_data,
    plot_results,
    train,
)

total_error_budget = 0.1
benchmarks_and_sizes = [("ae", [3, 4, 5, 6, 7, 8, 9, 10])]
data = generate_data(
    total_error_budget=total_error_budget,
    number_of_randomly_generated_distributions=1000,
    benchmarks_and_sizes=benchmarks_and_sizes,
)
model, x_test, y_test = train(data)
y_pred = model.predict(x_test)
product_diffs = evaluate(x_test, y_pred, total_error_budget)
product_diffs_dataset = evaluate(x_test, y_test, total_error_budget)
plot_results(product_diffs, product_diffs_dataset, legend=True, bin_width=4)
```

A more elaborate example of the implementation is shown in the [`example.ipynb`](https://github.com/munich-quantum-toolkit/problemsolver/blob/main/notebooks/resource_estimation/error_budget_optimization/example.ipynb) Jupyter notebook.

For more details, see {cite:p}`forster2025error_budget_optimization`.

## Quantum Circuit Optimization for the Fault-Tolerance Era: Do We Have to Start from Scratch?

Translating quantum circuits into a device's native gate set often increases gate count, amplifying noise in today's error-prone Noisy Intermediate-Scale Quantum (NISQ) devices.
Although optimizations exist to reduce gate counts, scaling to larger qubit and circuit sizes will see hardware errors dominate, blocking industrial-scale use. Error correction can enable Fault-Tolerant Quantum Computing (FTQC) but demands massive qubit overheads, often tens of thousands for small problems.

This motivates FTQC-oriented optimization techniques and raises the question: can NISQ techniques be adapted, or must new ones be developed?
We address this question by evaluating Qiskit and TKET optimization passes on benchmark circuits from MQT Bench.
As tools to directly design and evaluate fault-tolerant quantum circuit instances, we use resource estimation to assess FTQC requirements.

An example usage of the implementation is shown in the [`example.ipynb`](https://github.com/munich-quantum-toolkit/problemsolver/blob/main/notebooks/resource_estimation/fault_tolerant_optimization/example.ipynb) Jupyter notebook.

For more details, see {cite:p}`forster2025ft_circuit_optimization`.

```{toctree}
:hidden:

self
```

```{toctree}
:caption: User Guide
:glob:
:hidden:
:maxdepth: 1

references
```

```{toctree}
:caption: Developers
:glob:
:hidden:
:maxdepth: 1

contributing
support
```

```{toctree}
:caption: Python API Reference
:glob:
:hidden:
:maxdepth: 6

api/mqt/problemsolver/index
```

## Contributors and Supporters

The _[Munich Quantum Toolkit (MQT)](https://mqt.readthedocs.io)_ is developed by the [Chair for Design Automation](https://www.cda.cit.tum.de/) at the [Technical University of Munich](https://www.tum.de/) and supported by the [Munich Quantum Software Company (MQSC)](https://munichquantum.software).
Among others, it is part of the [Munich Quantum Software Stack (MQSS)](https://www.munich-quantum-valley.de/research/research-areas/mqss) ecosystem, which is being developed as part of the [Munich Quantum Valley (MQV)](https://www.munich-quantum-valley.de) initiative.

<div style="margin-top: 0.5em">
<div class="only-light" align="center">
  <img src="https://raw.githubusercontent.com/munich-quantum-toolkit/.github/refs/heads/main/docs/_static/mqt-logo-banner-light.svg" width="90%" alt="MQT Banner">
</div>
<div class="only-dark" align="center">
  <img src="https://raw.githubusercontent.com/munich-quantum-toolkit/.github/refs/heads/main/docs/_static/mqt-logo-banner-dark.svg" width="90%" alt="MQT Banner">
</div>
</div>

Thank you to all the contributors who have helped make MQT ProblemSolver a reality!

<p align="center">
<a href="https://github.com/munich-quantum-toolkit/problemsolver/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=munich-quantum-toolkit/problemsolver" />
</a>
</p>

The MQT will remain free, open-source, and permissively licensed—now and in the future.
We are firmly committed to keeping it open and actively maintained for the quantum computing community.

To support this endeavor, please consider:

- Starring and sharing our repositories: [https://github.com/munich-quantum-toolkit](https://github.com/munich-quantum-toolkit)
- Contributing code, documentation, tests, or examples via issues and pull requests
- Citing the MQT in your publications (see {doc}`References <references>`)
- Using the MQT in research and teaching, and sharing feedback and use cases
- Sponsoring us on GitHub: [https://github.com/sponsors/munich-quantum-toolkit](https://github.com/sponsors/munich-quantum-toolkit)

<p align="center">
<iframe src="https://github.com/sponsors/munich-quantum-toolkit/button" title="Sponsor munich-quantum-toolkit" height="32" width="114" style="border: 0; border-radius: 6px;"></iframe>
</p>
