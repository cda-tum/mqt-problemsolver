from __future__ import annotations

import itertools


def test_3(x: list[int]) -> int:
    return -x[0] - x[1] - x[2] + x[0] * x[1] + x[0] * x[2] + x[1] * x[2] - x[0] * x[1] * x[2]


def test_4(x: list[int]) -> int:
    return (
        -x[0]
        - x[1]
        - x[2]
        - x[3]
        + x[0] * x[1]
        + x[0] * x[2]
        + x[0] * x[3]
        + x[1] * x[2]
        + x[1] * x[3]
        + x[2] * x[3]
        - x[0] * x[1] * x[2]
        - x[0] * x[1] * x[3]
        - x[0] * x[2] * x[3]
        - x[1] * x[2] * x[3]
        + x[0] * x[1] * x[2] * x[3]
    )


def test_x_exactly(x: list[int]) -> int:
    return 1 - sum(x) + sum([2 * x[i] * x[j] for i in range(len(x)) for j in range(i + 1, len(x))])


# def test_x_min(x: list[int]) -> int:
#    return 1 - sum(x) + sum([x[i] * x[j] for i in range(len(x)) for j in range(i + 1, len(x))])


def test_x_max(x: list[int]) -> int:
    return sum([x[i] * x[j] for i in range(len(x)) for j in range(i + 1, len(x))])


def test_x_min(x: list[int]) -> int:
    return (
        -(sum(x) ** 2)
        + sum(x)
        + 2 * sum([x[i] * x[j] for i in range(len(x)) for j in range(i + 1, len(x))])
        # + sum([(1 - x[i]) * x[j] for i in range(len(x)) for j in range(i + 1, len(x)) if i != j])
        + sum([(1 - x[i]) * (1 - x[j]) for i in range(len(x)) for j in range(len(x)) if i != j])
    )


def all_encodings(size: int) -> list[list[int]]:
    return [list(i) for i in itertools.product([0, 1], repeat=size)]


def main() -> None:
    for encoding in all_encodings(2):
        print("\t".join([str(i) for i in encoding]), end="\t\t")
        print(test_x_min(encoding))


if __name__ == "__main__":
    main()
