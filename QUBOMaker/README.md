[![PyPI](https://img.shields.io/pypi/v/mqt.qubomaker?logo=pypi&style=flat-square)](https://pypi.org/project/mqt.qubomaker/)
![OS](https://img.shields.io/badge/os-linux%20%7C%20macos%20%7C%20windows-blue?style=flat-square)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg?style=flat-square)](https://opensource.org/licenses/MIT)
[![CI](https://img.shields.io/github/actions/workflow/status/cda-tum/mqt-qubomaker/ci.yml?branch=main&style=flat-square&logo=github&label=ci)](https://github.com/cda-tum/mqt-qubomaker/actions/workflows/ci.yml)
[![CD](https://img.shields.io/github/actions/workflow/status/cda-tum/mqt-qubomaker/cd.yml?style=flat-square&logo=github&label=cd)](https://github.com/cda-tum/mqt-qubomaker/actions/workflows/cd.yml)
[![Documentation](https://img.shields.io/readthedocs/mqt-qubomaker?logo=readthedocs&style=flat-square)](https://mqt.readthedocs.io/projects/qubomaker)
[![codecov](https://img.shields.io/codecov/c/github/cda-tum/mqt-qubomaker?style=flat-square&logo=codecov)](https://codecov.io/gh/cda-tum/mqt-qubomaker)

<p align="center">
<a href="https://mqt.readthedocs.io">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/cda-tum/mqt/main/docs/_static/mqt_light.png" width="60%">
  <img src="https://raw.githubusercontent.com/cda-tum/mqt/main/docs/_static/mqt_dark.png" width="60%">
</picture>
</a>
</p>

# MQT QUBOMaker: Automatic Generation of QUBO Formulations from Optimization Problem Specifications

MQT QUBOMaker is a framework that can be used to automatically generate QUBO formulations for various optimization problems based on a selection of constraints that define the problem.
It is developed by the [Chair for Design Automation](https://www.cda.cit.tum.de/) at the [Technical University of Munich](https://www.tum.de/) as part of the _[Munich Quantum Toolkit](https://mqt.readthedocs.io/) (MQT)_.

The tool allows users to create QUBO formulations, and, thus, interact with quantum algorithms, without requiring any background knowledge in the field of quantum computing. End-users can stay entirely within their domain of expertise while being shielded from the complex and error-prone mathematical tasks of QUBO reformulation.

Furthermore, MQT QUBOMaker supports a variety of different encodings. End users can easily switch between the encodings for evaluation purposes without any additional effort, a task that would otherwise require a large amount of tedious mathematical reformulation.

Currently, MQT QUBOMaker provides the following submodule:

- [_Pathfinder_](./src/mqt/qubomaker/pathfinder/README.md): This submodule provides a specialization of the QUBOMaker class for the solution of optimization problems involving the search for paths in a directed graph. It provides a large set of pathfinding-related constraints that are used to define individual problem instances.

The _Pathfinder_ submodule also has a supporting [GUI](https://cda-tum.github.io/mqt-qubomaker/) to further facilitate its use.

For more details, please refer to:

<p align="center">
  <a href="https://mqt-qubomaker.readthedocs.io/en/latest/">
  <img width=30% src="https://img.shields.io/badge/documentation-blue?style=for-the-badge&logo=read%20the%20docs" alt="Documentation" />
  </a>
</p>

If you have any questions, feel free to create a [discussion](https://github.com/cda-tum/mqt-qubomaker/discussions) or an [issue](https://github.com/cda-tum/mqt-qubomaker/issues) on [GitHub](https://github.com/cda-tum/mqt-qubomaker).

## Getting Started

`mqt-qubomaker` is available via [PyPI](https://pypi.org/project/mqt.qubomaker/).

```console
(venv) $ pip install mqt.qubomaker
```

The following code gives an example of the usage with the `pathfinder` submodule:

```python3
import mqt.qubomaker as qm
import mqt.qubomaker.pathfinder as pf

# define an example graph to investigate.
graph = qm.Graph.from_adjacency_matrix(
    [
        [0, 1, 3, 4],
        [2, 0, 4, 2],
        [1, 5, 0, 3],
        [3, 8, 1, 0],
    ]
)

# select the settings for the QUBO formulation.
settings = pf.PathFindingQUBOGeneratorSettings(
    encoding_type=pf.EncodingType.ONE_HOT, n_paths=1, max_path_length=4, loops=True
)

# define the generator to be used for the QUBO formulation.
generator = pf.PathFindingQUBOGenerator(
    objective_function=pf.MinimizePathLength(path_ids=[1]),
    graph=graph,
    settings=settings,
)

# add the constraints that define the problem instance.
generator.add_constraint(pf.PathIsValid(path_ids=[1]))
generator.add_constraint(
    pf.PathContainsVerticesExactlyOnce(vertex_ids=graph.all_vertices, path_ids=[1])
)

# generate and view the QUBO formulation as a QUBO matrix.
print(generator.construct_qubo_matrix())
```

**Detailed documentation and examples are available at [ReadTheDocs](https://mqt-qubomaker.readthedocs.io/en/latest/).**

## References

MQT QUBOMaker has been developed based on methods proposed in the following paper:

- D. Rovara, N. Quetschlich, and R. Wille "[A Framework to Formulate
  Pathfinding Problems for Quantum Computing](https://arxiv.org/abs/2404.10820)", arXiv, 2024

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
<picture>
<img src="https://raw.githubusercontent.com/cda-tum/mqt/main/docs/_static/logo-mqv.svg" width="28%" alt="MQV Logo">
</picture>
</p>
