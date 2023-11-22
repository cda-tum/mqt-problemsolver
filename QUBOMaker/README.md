# Pathfinder

This tool is meant to help prepare the use of VQAs to solve pathfinding problems on directed graphs, by employing a set of constraints. The following constraints are currently supported (red expressions are not QUBO):

- _Throughout this, we assume $`x_{\pi, v, i} = 0`$_ for $`v > |V|`$.
- _In cases where we want to work with loops. we assume $`x_{\pi,v,i + N} = x*{\pi,v,i}`$, otherwise $`x*{\pi,v,j} = 0`$ for $`j > N`$\_
  - `PathIsValid`
  - `MinimisePathLength`
  - `PathContainsEdgeExactlyOnce`
  - `PathsShareNoEdges`
- _Paths cannot be shorter than $`N`$. If there are, repeat one of the vertices to pad the length._
- _As a consequence, the adjacency matrix must have $`A_{ii} = 0`$ for each $`i`$.\_
- _In **unary** encoding, we assume $`x_{\pi, 1, i} = 1`$ for all $`\pi, i`$.

## `PathIsValid`

_Ensure that $`(u \rightarrow v) \in E`$ for each $`(u \rightarrow v) \in \pi`$ and each position holds a vertex_

- One-Hot: $$\sum_{(u \rightarrow v) \not \in E} \sum_{i = 1}^{N}x_{\pi, u, i}x_{\pi, v, i+1} + \sum_{i=1}^N \left(1-\sum_{v \in V}x_{\pi,v,i} \right)^2$$
- Unary: $$\sum_{(u \rightarrow v) \not \in E}\sum_{i=1}^{N} (x_{\pi,u,i}-x_{\pi,u+1,i})(x_{\pi,v,i+1}-x_{\pi,v+1,i+1})$$
- Binary

## `MinimisePathLength`

_Minimise $`\sum_{(u \rightarrow v) \in \pi} A*{uv}`$*

- One-Hot: $$\sum_{(u \rightarrow v) \in E} \sum_{i = 1}^{N} A_{uv}x_{\pi, u, i}x_{\pi, v, i+1}$$
- Unary: $$\sum_{(u \rightarrow v) \in E}\sum_{i=1}^{N} A_{uv}(x_{\pi,u,i}-x_{\pi,u+1,i})(x_{\pi,v,i+1}-x_{\pi,v+1,i+1})$$
- Binary

## `PathPositionIs`

_Given set of vertices $`V' \subseteq V`$, position $`i`$: ensure $`\pi_i \in v`$_

- One-Hot: $$1 - \sum_{v \in V'} x_{\pi, v, i}$$
- Unary: $$1 - \sum_{v\in V'}(x_{\pi, v, i} - x_{\pi, v + 1, i})$$
- Binary

## `PathContainsVertexExactlyOnce`

_Given $`v`$, ensure that: $`\left| \{i: \pi_i = v \} \right| = 1`$_

- One-Hot: $$\left( 1 - \sum_{i = 1}^N x_{\pi, v, i} \right) ^2$$
- Unary: $$\left( 1 - \sum_{i=1}^N (x_{\pi,v,i} - x_{\pi,v+1,i}) \right)^2$$
- Binary

## `PathContainsVertexAtLeastOnce`

_Given $`v`$, ensure that: $`\left| \{i: \pi_i = v \} \right| \geq 1`$_

- One-Hot: $$???$$
- Unary: $$???$$
- Binary

## `PathContainsVertexAtMostOnce`

_Given $`v`$, ensure that: $`\left| \{i: \pi_i = v \} \right| \leq 1`$_

- One-Hot: $$???$$
- Unary: $$???$$
- Binary

## `PathContainsEdgeExactlyOnce`

_Given $`e = (u \rightarrow v)`$, ensure that: $`|\{(i, i + 1) : \pi_i = u \wedge \pi_{i+1} = v\}| = 1`$\_

- One-Hot: $$\color{red} \left( 1 - \sum_{i=1}^{N}x_{\pi, u, i}x_{\pi, v, i + 1} \right)^2$$
- Unary: $$\color{red} \left( 1 - \sum_{i=1}^{N}(x_{\pi,u,i}-x_{\pi,u+1,i})(x_{\pi,v,i+1}-x_{\pi,v+1,i+1}) \right)^2$$
- Binary

## `PathContainsEdgeAtMostOnce`

_Given $`e = (u \rightarrow v)`$, ensure that: $`|\{(i, i + 1) : \pi_i = u \wedge \pi_{i+1} = v\}| \leq 1`$\_

- One-Hot: $$???$$
- Unary: $$???$$
- Binary

## `PathContainsEdgeAtLeastOnce`

_Given $`e = (u \rightarrow v)`$, ensure that: $`\left| \{(i, i + 1) : \pi_i = u \wedge \pi_{i+1} = v\} \right| \geq 1`$\_

- One-Hot: $$???$$
- Unary: $$???$$
- Binary

## `PathsShareNoVertices`

_Given two paths $`\pi^{(1)}`$ and $`\pi^{(2)}`$, $`\pi^{(1)}_V \cap \pi^{(2)}_V = \emptyset`$_

- One-Hot: $$\sum_{v \in V} \left[ \left(\sum_{i=1}^N x_{\pi^{(1)}, v, i} \right) \left(\sum_{i=1}^N x_{\pi^{(2)}, v, i} \right) \right]$$
- Unary: $$\sum_{v \in V} \left[ \left(\sum_{i=1}^N x_{\pi^{(1)},v,i} - x_{\pi^{(1)},v+1,i} \right) \left(\sum_{i=1}^N x_{\pi^{(2)},v,i} - x_{\pi^{(2)},v+1,i} \right) \right]$$
- Binary

## `PathsShareNoEdges`

_Given two paths $`\pi^{(1)}`$ and $`\pi^{(2)}`$, $`\pi^{(1)}_E \cap \pi^{(2)}_E = \emptyset`$_

- One-Hot: $$\color{red} \sum_{(u \rightarrow v) \in V} \left[ \left(\sum_{i=1}^{N} x_{\pi^{(1)}, u, i} x_{\pi^{(1)}, v, i + 1} \right) \left(\sum_{i=1}^{N} x_{\pi^{(2)}, u, i} x_{\pi^{(2)}, v, i + 1} \right) \right]$$
- Unary: $$\color{red} \sum_{(u \rightarrow v) \in V} \left[ \left(\sum_{i=1}^{N} (x_{\pi^{(1)},u,i}-x_{\pi^{(1)}u+1,i})(x_{\pi^{(1)}v,i+1}-x_{\pi^{(1)}v+1,i+1}) \right) \left(\sum_{i=1}^{N} (x_{\pi^{(2)},u,i}-x_{\pi^{(2)}u+1,i})(x_{\pi^{(2)}v,i+1}-x_{\pi^{(2)}v+1,i+1}) \right) \right]$$
- Binary

## `PrecedenceConstraint`

_Given a pair $`(u, v)`$, $`v`$ may not appear before $`u`$_

- One-Hot: $$???$$
- Unary: $$???$$
- Binary

---

---
