# MQT ProblemSolver

Two case studies:

# Kakuro Puzzle

First, the package must be installed:

```console
(venv) $ pip install mqt.predictor
```

# Travelling Salesman Problem (TSP)

```python
from mqt.predictor.driver import Predictor

predictor = Predictor()
prediction_index = predictor.predict("qasm_file_path")
```

# Repository Structure

```
test
```

# Reference

In case you are using MQT ProblemSolver in your work, we would be thankful if you referred to it by citing the following publication:

```bibtex
@misc{xxx,
  title={xxx},
  author={Quetschlich, Nils and Burgholzer, Lukas and Wille, Robert},
  year={2022},
}
```
