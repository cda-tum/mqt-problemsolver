[![PyPI](https://img.shields.io/pypi/v/mqt.problemsolver?logo=pypi&style=flat-square)](https://pypi.org/project/mqt.problemsolver/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg?style=flat-square)](https://opensource.org/licenses/MIT)
[![codecov](https://img.shields.io/codecov/c/github/munich-quantum-toolkit/problemsolver?style=flat-square&logo=codecov)](https://codecov.io/gh/munich-quantum-toolkit/problemsolver)

<p align="center">
  <a href="https://mqt.readthedocs.io">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/munich-quantum-toolkit/.github/refs/heads/main/docs/_static/logo-mqt-dark.svg" width="60%">
      <img src="https://raw.githubusercontent.com/munich-quantum-toolkit/.github/refs/heads/main/docs/_static/logo-mqt-light.svg" width="60%" alt="MQT Logo">
    </picture>
  </a>
</p>

# MQT ProblemSolver

MQT ProblemSolver provides a framework to utilize quantum computing as a technology for users with little to no quantum computing knowledge
It is developed as part of the [_Munich Quantum Toolkit (MQT)_](https://mqt.readthedocs.io).

<p align="center">
  <a href="https://mqt.readthedocs.io/projects/core">
  <img width=30% src="https://img.shields.io/badge/documentation-blue?style=for-the-badge&logo=read%20the%20docs" alt="Documentation" />
  </a>
</p>

## Key Features

- Progress toward an automated framework for solving optimization and constraint satisfaction problems with minimal quantum expertise
- Hybrid classical–quantum approaches for domain-specific applications such as satellite mission planning
- Methods for reducing compilation time through pre-compilation and optimizing quantum circuits
- Equivalence checking of classical circuits with quantum computing
- Utilizing resource estimation for evaluating and optimizing hardware requirements regarding fault-tolerant quantum computing

If you have any questions, feel free to create a [discussion](https://github.com/munich-quantum-toolkit/problemsolver/discussions) or an [issue](https://github.com/munich-quantum-toolkit/problemsolver/issues) on [GitHub](https://github.com/munich-quantum-toolkit/problemsolver).

## Contributors and Supporters

The _[Munich Quantum Toolkit (MQT)](https://mqt.readthedocs.io)_ is developed by the [Chair for Design Automation](https://www.cda.cit.tum.de/) at the [Technical University of Munich](https://www.tum.de/) and supported by the [Munich Quantum Software Company (MQSC)](https://munichquantum.software).
Among others, it is part of the [Munich Quantum Software Stack (MQSS)](https://www.munich-quantum-valley.de/research/research-areas/mqss) ecosystem, which is being developed as part of the [Munich Quantum Valley (MQV)](https://www.munich-quantum-valley.de) initiative.

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/munich-quantum-toolkit/.github/refs/heads/main/docs/_static/mqt-logo-banner-dark.svg" width="90%">
    <img src="https://raw.githubusercontent.com/munich-quantum-toolkit/.github/refs/heads/main/docs/_static/mqt-logo-banner-light.svg" width="90%" alt="MQT Partner Logos">
  </picture>
</p>

Thank you to all the contributors who have helped make MQT ProblemSolver a reality!

<p align="center">
  <a href="https://github.com/munich-quantum-toolkit/core/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=munich-quantum-toolkit/problemsolver" alt="Contributors to munich-quantum-toolkit/problemsolver" />
  </a>
</p>

The MQT will remain free, open-source, and permissively licensed—now and in the future.
We are firmly committed to keeping it open and actively maintained for the quantum computing community.

To support this endeavor, please consider:

- Starring and sharing our repositories: https://github.com/munich-quantum-toolkit
- Contributing code, documentation, tests, or examples via issues and pull requests
- Citing the MQT in your publications (see [Cite This](#cite-this))
- Citing our research in your publications (see [References](https://mqt.readthedocs.io/projects/problemsolver/en/latest/references.html))
- Using the MQT in research and teaching, and sharing feedback and use cases
- Sponsoring us on GitHub: https://github.com/sponsors/munich-quantum-toolkit

<p align="center">
  <a href="https://github.com/sponsors/munich-quantum-toolkit">
  <img width=20% src="https://img.shields.io/badge/Sponsor-white?style=for-the-badge&logo=githubsponsors&labelColor=black&color=blue" alt="Sponsor the MQT" />
  </a>
</p>

## Getting Started

`mqt.problemsolver` is available via [PyPI](https://pypi.org/project/mqt.problemsolver/).

```console
(.venv) $ pip install mqt.problemsolver
```

**Detailed documentation and examples are available at [ReadTheDocs](https://mqt.readthedocs.io/projects/problemsolver).**

## System Requirements

MQT ProblemSolver can be installed on all major operating systems with all supported Python versions.
Building (and running) is continuously tested under Linux, macOS, and Windows using the [latest available system versions for GitHub Actions](https://github.com/actions/runner-images).

## Cite This

Please cite the work that best fits your use case.

### MQT ProblemSolver (the tool)

When citing the software itself or results produced with it, cite the MQT ProblemSolver paper:

```bibtex
@inproceedings{quetschlich2023mqtproblemsolver,
  title        = {{Towards an Automated Framework for Realizing Quantum Computing Solutions}},
  author       = {Quetschlich, Nils and Burgholzer, Lukas and Wille, Robert},
  year         = 2023,
  booktitle    = {International Symposium on Multiple-Valued Logic (ISMVL)},
  eprint       = {2210.14928},
  eprinttype   = {arXiv}
}
```

### The Munich Quantum Toolkit (the project)

When discussing the overall MQT project or its ecosystem, cite the MQT Handbook:

```bibtex
@inproceedings{mqt,
  title        = {The {{MQT}} Handbook: {{A}} Summary of Design Automation Tools and Software for Quantum Computing},
  shorttitle   = {{The MQT Handbook}},
  author       = {Wille, Robert and Berent, Lucas and Forster, Tobias and Kunasaikaran, Jagatheesan and Mato, Kevin and Peham, Tom and Quetschlich, Nils and Rovara, Damian and Sander, Aaron and Schmid, Ludwig and Schoenberger, Daniel and Stade, Yannick and Burgholzer, Lukas},
  year         = 2024,
  booktitle    = {IEEE International Conference on Quantum Software (QSW)},
  doi          = {10.1109/QSW62656.2024.00013},
  eprint       = {2405.17543},
  eprinttype   = {arxiv},
  addendum     = {A live version of this document is available at \url{https://mqt.readthedocs.io}}
}
```

### Peer-Reviewed Research

When citing the underlying methods and research, please reference the most relevant peer-reviewed publications from the list below:

[[1]](https://arxiv.org/pdf/2210.14928.pdf)
N. Quetschlich and L. Burgholzer and R. Wille. Towards an Automated Framework for Realizing Quantum Computing Solutions.
_International Symposium on Multiple-Valued Logic (ISMVL)_, 2023.

[[2]](https://arxiv.org/pdf/2308.00029.pdf)
N. Quetschlich, V. Koch, L. Burgholzer, and R. Wille. A Hybrid Classical Quantum Computing Approach to the Satellite Mission Planning Problem.
_IEEE International Conference on Quantum Computing and Engineering (QCE)_, 2023.

[[3]](https://arxiv.org/pdf/2305.04941.pdf)
N. Quetschlich, L. Burgholzer, and R. Wille. Reducing the Compilation Time of Quantum Circuits Using Pre-Compilation on the Gate Level.
_IEEE International Conference on Quantum Computing and Engineering (QCE)_, 2023.

[[4]](https://arxiv.org/pdf/2402.12434.pdf)
N. Quetschlich, M. Soeken, P. Murali, and R. Wille. Reducing the Compilation Time of Quantum Circuits Using Pre-Compilation on the Gate Level.
_IEEE International Conference on Quantum Computing and Engineering (QCE)_, 2024.

[[5]](https://arxiv.org/pdf/2408.14539.pdf)
N. Quetschlich, T. Forster, A. Osterwind, D. Helms, and R. Wille. Towards Equivalence Checking of Classical Circuits Using Quantum Computing.
_IEEE International Conference on Quantum Computing and Engineering (QCE)_, 2024.

[6]
T. Forster, N. Quetschlich, M. Soeken, and R. Wille. Improving Hardware Requirements for Fault-Tolerant Quantum Computing by Optimizing Error Budget Distributions.
_IEEE International Conference on Quantum Computing and Engineering (QCE)_, 2025.

[7]
T. Forster, N. Quetschlich, and R. Wille. Quantum Circuit Optimization for the Fault-Tolerance Era: Do We Have to Start from Scratch?
_IEEE International Conference on Quantum Computing and Engineering (QCE)_, 2025.

---

## Acknowledgements

The Munich Quantum Toolkit has been supported by the European Research Council (ERC) under the European Union's Horizon 2020 research and innovation program (grant agreement No. 101001318), the Bavarian State Ministry for Science and Arts through the Distinguished Professorship Program, as well as the Munich Quantum Valley, which is supported by the Bavarian state government with funds from the Hightech Agenda Bayern Plus.

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/munich-quantum-toolkit/.github/refs/heads/main/docs/_static/mqt-funding-footer-dark.svg" width="90%">
    <img src="https://raw.githubusercontent.com/munich-quantum-toolkit/.github/refs/heads/main/docs/_static/mqt-funding-footer-light.svg" width="90%" alt="MQT Funding Footer">
  </picture>
</p>
