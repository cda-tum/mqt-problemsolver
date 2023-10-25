from cost_functions import CostFunction, MinimisePathLength, PathContainsVerticesAtLeastOnce, PathContainsVerticesAtMostOnce, PathContainsVerticesExactlyOnce, PathEndsAt, PathIsLoop, PathIsValid, PathPositionIs, PathStartsAt, PathsShareNoEdges, PathsShareNoVertices, merge
from graph import Graph
import inspect

def main() -> None:
    cost_functions: list[CostFunction] = []
    optimisation_goals: list[CostFunction] = []
    
    with open("graph", "r") as file:
        graph = Graph.read(file)
    
    while next := prompt_new_cost_function():
        if isinstance(next, MinimisePathLength):
            optimisation_goals.append(next)
        else:
            cost_functions.append(next)
    
    if not optimisation_goals:
        print("You did not include an optimisation goal. Add one now? [Y/n]")
        x = input()
        if x.lower() != "n":
            optimisation_goals.append(MinimisePathLength([0]))
    
    hamiltonian = merge(cost_functions, optimisation_goals)
    print(f"The final merged cost function is:\n{hamiltonian}")
    
    #TODO actually finalise and use this cost function
    
def prompt_new_cost_function() -> CostFunction | None:
    x = PathPositionIs
    functions = [
        ("Position [i] is [v]", "True if the i-th vertex in the path ist the given vertex.", PathPositionIs),
        ("Starts at [v]", "True if the path starts at the given vertex.", PathStartsAt),
        ("Ends at [v]", "True if the path ends at the given vertex.", PathEndsAt),
        ("Contains [v] exactly once", "True if the contains the given vertex exactly once.", PathContainsVerticesExactlyOnce),
        ("Contains [v] at least once", "True if the contains the given vertex at least once.", PathContainsVerticesAtLeastOnce),
        ("Contains [v] at most once", "True if the contains the given vertex at most once.", PathContainsVerticesAtMostOnce),
        ("Is loop", "True if the path ends at the same vertex it starts at.", PathIsLoop),
        ("Paths share no vertices", "True if the given paths do not share any vertices.", PathsShareNoVertices),
        ("Paths share no edge", "True if the give paths do not share any edges.", PathsShareNoEdges),
        ("Path [p] is valid", "True if the given path is valid", PathIsValid),
        ("Minimise path [p] length", "Optimisation goal. Keep the length of the path minimal.", MinimisePathLength)
    ]
    for i, (name, description, function) in enumerate(functions):
        print(f"{i + 1}\t| {name} {' ' * (30 - len(name))}| {description}")
    print("Type the number of the cost function you would like to add, or leave empty to finish adding cost functions")
    n = input()
    try:
        n = int(n)
        if n > 0 and n <= len(functions):
            f = functions[n - 1][2]
            n_args = len(inspect.signature(f.__init__).parameters) - 1
            args = [None for _ in range(n_args)]
            return f(*args)
        else:
            return None
    except ValueError:
        return None

if __name__ == "__main__":
    main()