# Pathfinder

This tool is meant to help prepare the use of VQAs to solve pathfinding problems on directed graphs, by employing a set of constraints. The following constraints are currently supported (red expressions are not QUBO):

## `PathIsValid`
Ensure that $`(u \rightarrow v) \in E`$ for each $`(u \rightarrow v) \in \pi`$ 
- One-Hot: $$\sum_{(u \rightarrow v) \not \in E} \sum_{i = 1}^{N - 1}x_{\pi, u, i}x_{\pi, v, i+1}$$
- Unary
- Binary

## `MinimisePathLength`
Minimise $`\sum_{(u \rightarrow v) \in \pi} A_{uv}`$ 
- One-Hot: $$\sum_{(u \rightarrow v) \in E} \sum_{i = 1}^{N - 1} A_{uv}x_{\pi, u, i}x_{\pi, v, i+1}$$
	- (loop): add $`A_{uv}x_{\pi, u, N} x_{\pi, v, 1}`$ in outer sum
- Unary
- Binary

## `PathPositionIs`
Given vertex $`v`$, position $`i`$: ensure $`\pi_i = v`$ 
- One-Hot: $$1 - x_{\pi, v, i}$$
- Unary: $$1 - x_{\pi, v, i} \left( 1 - x_{\pi, v + 1, i} \right)$$
- Binary

## `PathContainsVertexExactlyOnce`
Given $`v`$, ensure that: $`\left| \{i: \pi_i = v \} \right| = 1`$ 
- One-Hot: $$\left( 1 - \sum_{i = 1}^N x_{\pi, v, i} \right) ^2$$
- Unary
- Binary

## `PathContainsVertexAtLeastOnce`
Given $`v`$, ensure that: $`\left| \{i: \pi_i = v \} \right| \geq 1`$ 
- One-Hot: $$???$$
- Unary
- Binary

## `PathContainsVertexAtMostOnce`
Given $`v`$, ensure that: $`\left| \{i: \pi_i = v \} \right| \leq 1`$ 
- One-Hot: $$???$$
- Unary
- Binary

## `PathContainsEdgeExactlyOnce`
Given $`e = (u \rightarrow v)`$, ensure that: $`|\{(i, i + 1) : \pi_i = u \wedge \pi_{i+1} = v\}| = 1`$ 
- One-Hot: $$\color{red} \left( 1 - \sum_{i=1}^{N-1}x_{\pi, u, i}x_{\pi, v, i + 1} \right)^2$$
	- (loop): add $`x_{\pi, u, N}x_{\pi, v, 1}`$ inside brackets
- Unary
- Binary

## `PathContainsEdgeAtMostOnce`
Given $`e = (u \rightarrow v)`$, ensure that: $`|\{(i, i + 1) : \pi_i = u \wedge \pi_{i+1} = v\}| \leq 1`$ 
- One-Hot: $$???$$
- Unary
- Binary

## `PathContainsEdgeAtLeastOnce`
Given $`e = (u \rightarrow v)`$, ensure that: $`\left| \{(i, i + 1) : \pi_i = u \wedge \pi_{i+1} \geq v\} \right| = 1`$ 
- One-Hot: $$???$$
- Unary
- Binary

## `PathsShareNoVertices`
Given two paths $`\pi^{(1)}`$ and $`\pi^{(2)}`$,  $`\pi^{(1)}_V \cap \pi^{(2)}_V = \emptyset`$ 
- One-Hot: $$\sum_{v \in V} \left[ \left(\sum_{i=1}^N x_{\pi^{(1)}, v, i} \right) \left(\sum_{i=1}^N x_{\pi^{(2)}, v, i} \right) \right]$$
- Unary
- Binary

## `PathsShareNoEdges`
Given two paths $`\pi^{(1)}`$ and $`\pi^{(2)}`$,  $`\pi^{(1)}_E \cap \pi^{(2)}_E = \emptyset`$ 
- One-Hot: $$\color{red} \sum_{(u \rightarrow v) \in V} \left[ \left(\sum_{i=1}^{N-1} x_{\pi^{(1)}, u, i} x_{\pi^{(1)}, v, i + 1} \right) \left(\sum_{i=1}^{N-1} x_{\pi^{(2)}, u, i} x_{\pi^{(2)}, v, i + 1} \right) \right]$$
	- (loop) add $`\color{red} \left( x_{\pi^{(1)}, u, N} x_{\pi^{(1)}, v, 1} \right)\left( x_{\pi^{(2)}, u, N} x_{\pi^{(2)}, v, 1} \right)$
- Unary
- Binary

## `PrecedenceConstraint`
Given a pair $`(u, v)`$, $`v`$ may not appear before $`u`$ 
- One-Hot: $$\sum_{i=1}^N \left[ x_{\pi, v, i} \left( 1 - \sum_{j=1}^{i-1} x_{\pi, u, j} \right) \right]$$
	- TODO: this doesn't work if the same vertex may appear multiple times
- Unary
- Binary
---
----

