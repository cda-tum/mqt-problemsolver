Encoding Schemes
================

The *Pathfinder* submodule provides three different encoding schemes. The encoding scheme determines, how
many binary variables are required to represent the problem's cost function and how these variables are to be
interpreted.

An example of the same path represented in each of the three encoding schemes is shown below.

.. image:: ../_static/encodings.png
    :align: center
    :alt: Encoding schemes


For each encoding scheme, the function :math:`\delta(x, \pi^{(i)}, v, j)` is defined, returning 1 if vertex :math:`v` is located at position :math:`j` in path :math:`\pi^{(i)}`, and 0 otherwise.
The complexity of this function depends on the encoding scheme used.

Below, :math:`N` denotes the maximum length of a path, :math:`|V|` the number of vertices in the graph, and :math:`|\Pi|` the number of paths to be found.


One-Hot
-------

In the one-hot encoding scheme, :math:`N \cdot |V| \cdot |\Pi|` binary variables :math:`x_{v,j,\pi^{(i)}}` are
used to represent the problem.

An individual variable with value 1 indicates that the corresponding vertex :math:`v` is located at position
:math:`j` in path :math:`\pi^{(i)}`. Assignments such that :math:`x_{v,j,\pi^{(i)}} = 1` for more than one :math:`v` and the same :math:`j` and :math:`\pi^{(i)}``
are invalid.

This encoding scheme is very expressive, but also uses a large amount of binary variables. It is also
very sparse, meaning that there exists a large number of invalid assignments. Moving from one valid assignment
to another requires at least two bitflips.

$$\\delta(x, \\pi^{(i)}, v, j) = x_{v, j, \\pi^{(i)}}$$

Domain-Wall
-----------

In the domain-wall encoding scheme, :math:`N \cdot |V| \cdot |\Pi|` binary variables :math:`x_{v,j,\pi^{(i)}}` are used
to represent the problem.

For each position :math:`j` and path :math:`\pi^{(i)}`, the variables :math:`x_{v,j,\pi^{(i)}}` are read as a bitstring
:math:`\overline{x_{j,\pi^{(i)}}}` of length :math:`|V|`. If the first :math:`n` bits of this bitstring are 1, this indicates that
vertex :math:`v_n` is located at position :math:`j` in path :math:`\pi^{(i)}`.

Compared to the one-hot encoding, it is easier to move from one valid assignment to another, as only one bitflip
is required for that. However, there exist just as many invalid encodings.

$$\\delta(x, \\pi^{(i)}, v, j) = x_{v, j, \\pi^{(i)}} - x_{v + 1, j, \\pi^{(i)}}$$

Binary
------

In binary encoding :math:`N \cdot \text{log}(|V|) \cdot |\Pi|` binary variables :math:`x_{v,j,\pi^{(i)}}` are used
to represent the problem.

For each position :math:`j` and path :math:`\pi^{(i)}`, the variables :math:`x_{v,j,\pi^{(i)}}` are read as a bitstring
:math:`\overline{x_{j,\pi^{(i)}}}` of length :math:`\text{log}(|V|)`. This bitstring is interpreted as a binary number,
representing the index of the vertex located at position :math:`j` in path :math:`\pi^{(i)}`.

This is a dense encoding, as no invalid assignment exists. However, it is less expressive and more complex
than the other encodings. In particular, cost functions using the binary encoding are rarely of quadratic order,
and, therefore, often require additional auxiliary variables.

$$\\delta(x, \\pi^{(i)}, v, j) = \\prod_{w=1}^{\\text{log}_2(\|V\| + 1)} (\\text{b}(v, w) x_{w, j, \\pi^{(i)}}) + ((1 - \\text{b}(v, w)) (1 - x_{w, j, \\pi^{(i)}}))$$

*where* :math:`\text{b}(n, i)` *denotes the* :math:`i\text{-th}` *bit of the binary representation of* :math:`n` *.*
