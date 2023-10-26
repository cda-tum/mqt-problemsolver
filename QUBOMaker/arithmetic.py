# pylint: disable=too-many-arguments
# Due to the general structure of the visitor pattern, using larger methods makes sense.

from __future__ import annotations
from dataclasses import dataclass
import functools
from typing import Any, Callable, Generic, TypeVar
import sympy as sp
from sympy.core.expr import Expr

@dataclass
class ArithmeticItem:
    order: int

    def __init__(self, order: int) -> None:
        self.order = order

    @staticmethod
    def __item_or_constant(value: Any) -> ArithmeticItem:
        if isinstance(value, ArithmeticItem):
            return value
        if isinstance(value, (int, float)):
            return Constant(float(value))
        raise TypeError(type(value))

    def __add__(self, other: Any) -> ArithmeticItem:
        return Addition(self, ArithmeticItem.__item_or_constant(other))

    def __radd__(self, other: Any) -> ArithmeticItem:
        x: Addition = self + other
        return Addition(x.right, x.left)

    def __sub__(self, other: Any) -> ArithmeticItem:
        return self + -1 * other

    def __rsub__(self, other: Any) -> ArithmeticItem:
        return other + -1 * self

    def __mul__(self, other: Any) -> ArithmeticItem:
        return Multiplication(self, ArithmeticItem.__item_or_constant(other))

    def __rmul__(self, other: Any) -> ArithmeticItem:
        x: Multiplication = self * other
        return Multiplication(x.right, x.left)

    def __truediv__(self, other: Any) -> ArithmeticItem:
        return Division(self, ArithmeticItem.__item_or_constant(other))

    def __rtruediv__(self, other: Any) -> ArithmeticItem:
        return Division(ArithmeticItem.__item_or_constant(other), self)

    def __pow__(self, other: Any) -> ArithmeticItem:
        return Exponentiation(self, ArithmeticItem.__item_or_constant(other))

    def __rpow__(self, other: Any) -> ArithmeticItem:
        return Exponentiation(ArithmeticItem.__item_or_constant(other), self)

    def _repr_latex_(self) -> str:
        s = SimplifyingTransformer()
        l = LatexTransformer()
        lat = l.transform(s.transform(self))
        return "$$" + lat + "$$"
        #display(Math(lat))


@dataclass
class Constant(ArithmeticItem):
    __match_args__ = ("value",)
    value: float

    def __init__(self, value: float) -> None:
        super().__init__(0)
        self.value = int(value) if value == int(value) else value

    def __str__(self) -> str:
        return str(self.value)

@dataclass
class Variable(ArithmeticItem):
    __match_args__ = ("name", "subscripts", "superscripts")
    name: str
    subscripts: list[ArithmeticItem]
    superscripts: list[ArithmeticItem]

    def __init__(self, name: str, subscripts: list[ArithmeticItem | str | int] = None, superscripts: list[ArithmeticItem | str | int] = None) -> None:
        super().__init__(0)
        subscripts = [s if isinstance(s, ArithmeticItem) else Variable(s, [], []) if isinstance(s, str) else Constant(s) for s in subscripts] if subscripts else []
        superscripts = [s if isinstance(s, ArithmeticItem) else Variable(s, [], []) if isinstance(s, str) else Constant(s) for s in superscripts] if superscripts else []
        self.name = name
        self.subscripts = subscripts
        self.superscripts = superscripts

    def __str__(self) -> str:
        if not self.subscripts and not self.superscripts:
            return self.name
        return f"{self.name}{('_{' + ','.join([str(x) for x in self.subscripts]) + '}') if self.subscripts else ''}{('_{' + ','.join([str(x) for x in self.superscripts]) + '}') if self.superscripts else ''}"

@dataclass
class Addition(ArithmeticItem):
    __match_args__ = ("left", "right")
    left: ArithmeticItem
    right: ArithmeticItem

    def __init__(self, left: ArithmeticItem, right: ArithmeticItem) -> None:
        super().__init__(3)
        self.left = left
        self.right = right

    def __str__(self) -> str:
        return f"({self.left} + {self.right})"

@dataclass
class Multiplication(ArithmeticItem):
    __match_args__ = ("left", "right")
    left: ArithmeticItem
    right: ArithmeticItem

    def __init__(self, left: ArithmeticItem, right: ArithmeticItem) -> None:
        super().__init__(2)
        self.left = left
        self.right = right

    def __str__(self) -> str:
        return f"({self.left} * {self.right})"

@dataclass
class Division(ArithmeticItem):
    __match_args__ = ("left", "right")
    left: ArithmeticItem
    right: ArithmeticItem

    def __init__(self, left: ArithmeticItem, right: ArithmeticItem) -> None:
        super().__init__(2)
        self.left = left
        self.right = right

    def __str__(self) -> str:
        return f"({self.left} / {self.right})"

@dataclass
class Exponentiation(ArithmeticItem):
    __match_args__ = ("base", "exponent")
    base: ArithmeticItem
    exponent: ArithmeticItem

    def __init__(self, base: ArithmeticItem, exponent: ArithmeticItem) -> None:
        super().__init__(1)
        self.base = base
        self.exponent = exponent

    def __str__(self) -> str:
        return f"({self.base} ^ {self.exponent})"

@dataclass
class Sum(ArithmeticItem):
    expression: ArithmeticItem

    def __init__(self, expression: ArithmeticItem) -> None:
        super().__init__(3)
        self.expression = expression

@dataclass
class SumFromTo(Sum):
    __match_args__ = ("expression", "variable", "from_value", "to_value")
    variable: ArithmeticItem
    from_value: ArithmeticItem
    to_value: ArithmeticItem

    def __init__(self, expression: ArithmeticItem, variable: ArithmeticItem, from_value: ArithmeticItem, to_value: ArithmeticItem) -> None:
        super().__init__(expression)
        self.variable = variable
        self.from_value = from_value
        self.to_value = to_value

    def __str__(self) -> str:
        return f"SUM[{self.variable}={self.from_value};{self.to_value}]({self.expression})"

@dataclass
class SumSet(Sum):
    __match_args__ = ("expression", "variables", "set_expression", "set_callback")
    variables: list[ArithmeticItem]
    set_expression: str
    set_callback: Callable[[], list[ArithmeticItem | tuple[ArithmeticItem]]]

    def __init__(self, expression: ArithmeticItem, variables: list[ArithmeticItem], set_expression: str, set_callback: Callable[[], list[ArithmeticItem | tuple[ArithmeticItem]]]) -> None:
        super().__init__(expression)
        self.variables = variables
        self.set_expression = set_expression
        self.set_callback = set_callback

T = TypeVar('T')
class VisitingTransformer(Generic[T]):

    def transform(self, expression: ArithmeticItem, parent: ArithmeticItem | None = None) -> T:
        result: T = None
        match expression:
            case Constant(value): result = self.transform_constant(value, expression, parent)
            case Variable(name, subscripts, superscripts): result = self.transform_variable(name, subscripts, superscripts, expression, parent)
            case Addition(left, right): result = self.transform_addition(left, right, expression, parent)
            case Multiplication(left, right): result = self.transform_multiplication(left, right, expression, parent)
            case Division(left, right): result = self.transform_division(left, right, expression, parent)
            case Exponentiation(left, right): result = self.transform_exponentiation(left, right, expression, parent)
            case SumFromTo(expr, variable, from_value, to_value): result = self.transform_sum_from_to(expr, variable, from_value, to_value, expression, parent)
            case SumSet(expr, variable, from_value, set_callback): result = self.transform_sum_set(expr, variable, from_value, set_callback, expression, parent)
            case _: raise ValueError(f"{expression} ({parent})")

        return  result

    def transform_constant(self, _value: int, expression: Constant, _parent: ArithmeticItem) -> T:
        return expression
    def transform_variable(self, name: str, subscripts: list[ArithmeticItem], superscripts: list[ArithmeticItem], _expression: Variable, _parent: ArithmeticItem) -> T:
        return Variable(name, [self.transform(v) for v in subscripts], [self.transform(v) for v in superscripts])
    def transform_addition(self, left: ArithmeticItem, right: ArithmeticItem, expression: Addition, _parent: ArithmeticItem) -> T:
        return Addition(self.transform(left, expression), self.transform(right, expression))
    def transform_multiplication(self, left: ArithmeticItem, right: ArithmeticItem, expression: Multiplication, _parent: ArithmeticItem) -> T:
        return Multiplication(self.transform(left, expression), self.transform(right, expression))
    def transform_division(self, left: ArithmeticItem, right: ArithmeticItem, expression: Division, _parent: ArithmeticItem) -> T:
        return Division(self.transform(left, expression), self.transform(right, expression))
    def transform_exponentiation(self, base: ArithmeticItem, exponent: ArithmeticItem, expression: Exponentiation, _parent: ArithmeticItem) -> T:
        return Exponentiation(self.transform(base, expression), self.transform(exponent, expression))
    def transform_sum_from_to(self, child: ArithmeticItem, variable: ArithmeticItem, from_value: ArithmeticItem, to_value: ArithmeticItem, expression: SumFromTo, _parent: ArithmeticItem) -> T:
        return SumFromTo(self.transform(child, expression), self.transform(variable), self.transform(from_value), self.transform(to_value))
    def transform_sum_set(self, child: ArithmeticItem, variables: list[ArithmeticItem], set_expression: str, set_callback: Callable[[], list[ArithmeticItem | tuple[ArithmeticItem]]], expression: SumSet, _parent: ArithmeticItem) -> T:
        return SumSet(self.transform(child, expression), [self.transform(v) for v in variables], set_expression, set_callback)

class SimplifyingTransformer(VisitingTransformer[ArithmeticItem]):

    def transform_multiplication(self, left: ArithmeticItem, right: ArithmeticItem, expression: Multiplication, _parent: ArithmeticItem) -> ArithmeticItem:
        l = self.transform(left, expression)
        r = self.transform(right, expression)

        if l == Constant(1):
            return r
        if r == Constant(1):
            return l
        if l == Constant(0) or r == Constant(0):
            return Constant(0)

        def leftmost_is_constant(expr: ArithmeticItem) -> bool:
            match expr:
                case Constant(_ko): return True
                case Multiplication(left, _): return leftmost_is_constant(left)
                case _: return False

        if not leftmost_is_constant(l) and leftmost_is_constant(r):
            return Multiplication(r, l)

        return Multiplication(l, r)

    def transform_division(self, left: ArithmeticItem, right: ArithmeticItem, expression: Multiplication, _parent: ArithmeticItem) -> ArithmeticItem:
        l = self.transform(left, expression)
        r = self.transform(right, expression)

        if l == Constant(1):
            return r
        if r == Constant(1):
            return l

        return Division(l, r)

    def transform_addition(self, left: ArithmeticItem, right: ArithmeticItem, expression: Addition, _parent: ArithmeticItem) -> ArithmeticItem:
        l = self.transform(left, expression)
        r = self.transform(right, expression)

        if l == Constant(0):
            return r
        if r == Constant(0):
            return l

        return Addition(l, r)

    def transform_exponentiation(self, base: ArithmeticItem, exponent: ArithmeticItem, expression: Exponentiation, parent: ArithmeticItem) -> ArithmeticItem:
        base = self.transform(base, expression)
        exponent = self.transform(exponent, expression)

        if base == Constant(0):
            return Constant(0)
        if base == Constant(1):
            return Constant(1)
        if exponent == Constant(0):
            return Constant(1)
        if exponent == Constant(1):
            return base
        return super().transform_exponentiation(base, exponent, expression, parent)

class SympyTransformer(VisitingTransformer[sp.Expr]):

    def transform_addition(self, left: ArithmeticItem, right: ArithmeticItem, expression: Addition, _parent: ArithmeticItem) -> Expr:
        l = self.transform(left, expression)
        r = self.transform(right, expression)
        return l + r

    def transform_constant(self, value: int, expression: Constant, _parent: ArithmeticItem) -> Expr:
        return value

    def transform_division(self, left: ArithmeticItem, right: ArithmeticItem, expression: Division, _parent: ArithmeticItem) -> Expr:
        l = self.transform(left, expression)
        r = self.transform(right, expression)
        return l / r

    def transform_exponentiation(self, base: ArithmeticItem, exponent: ArithmeticItem, expression: Exponentiation, _parent: ArithmeticItem) -> Expr:
        l = self.transform(base, expression)
        r = self.transform(exponent, expression)
        return l ** r

    def transform_multiplication(self, left: ArithmeticItem, right: ArithmeticItem, expression: Multiplication, _parent: ArithmeticItem) -> Expr:
        l = self.transform(left, expression)
        r = self.transform(right, expression)
        return l * r

    def transform_variable(self, name: str, subscripts: list[ArithmeticItem], superscripts: list[ArithmeticItem], expression: Variable, _parent: ArithmeticItem) -> Expr:
        return sp.Symbol(str(expression))

class LatexTransformer(VisitingTransformer[str]):

    def transform(self, expression: ArithmeticItem, parent: ArithmeticItem | None = None) -> str:
        sub = super().transform(expression, parent)
        if parent and parent.order < expression.order:
            return f"\\left( {sub} \\right)"
        return sub

    def transform_constant(self, value: int, expression: Constant, _parent: ArithmeticItem) -> T:
        return str(int(value) if int(value) == value else value)

    def transform_variable(self, name: str, subscripts: list[ArithmeticItem], superscripts: list[ArithmeticItem], expression: Variable, _parent: ArithmeticItem) -> T:
        return f"{name}{'_{' + ''.join([self.transform(x) for x in subscripts]) + '}' if subscripts else ''}{'^{' + ''.join([self.transform(x) for x in superscripts]) + '}' if superscripts else ''}"

    def transform_addition(self, left: ArithmeticItem, right: ArithmeticItem, expression: Addition, _parent: ArithmeticItem) -> T:
        l = self.transform(left, expression)
        r = self.transform(right, expression)
        if r.startswith("-"):
            return f"{l} - {r[1:]}"
        return f"{l} + {r}"

    def transform_multiplication(self, left: ArithmeticItem, right: ArithmeticItem, expression: Multiplication, parent: ArithmeticItem) -> T:
        match left:
            case Constant(-1): return f"-{self.transform(right, expression)}" if parent.order > expression.order else f"(-{self.transform(right, expression)})"
            case Constant(1): return self.transform(right, expression)
            case _: return f"{self.transform(left, expression)} {self.transform(right, expression)}"

    def transform_division(self, left: ArithmeticItem, right: ArithmeticItem, expression: Division, _parent: ArithmeticItem) -> T:
        return f"\\frac{{{self.transform(left, expression)}}}{{{self.transform(right, expression)}}}"

    def transform_exponentiation(self, base: ArithmeticItem, exponent: ArithmeticItem, expression: Exponentiation, _parent: ArithmeticItem) -> T:
        if isinstance(base, Variable) and base.superscripts:
            return f"\\left ({self.transform(base, expression)} \\right )^{{{self.transform(exponent, expression)}}}"
        return f"{self.transform(base, expression)}^{{{self.transform(exponent, expression)}}}"

    def transform_sum_from_to(self, child: ArithmeticItem, variable: ArithmeticItem, from_value: ArithmeticItem, to_value: ArithmeticItem, expression: SumFromTo, _parent: ArithmeticItem) -> T:
        return f"\\sum_{{{self.transform(variable)}={self.transform(from_value, expression)}}}^{{{self.transform(to_value, expression)}}} {self.transform(child, expression)}"

    def transform_sum_set(self, child: ArithmeticItem, variables: list[ArithmeticItem], set_expression: str, set_callback: Callable[[], list[ArithmeticItem | tuple[ArithmeticItem]]], expression: SumSet, _parent: ArithmeticItem) -> T:
        return f"\\sum_{{{('(' if len(variables) > 1 else '') + ', '.join([self.transform(v, expression) for v in variables]) + (')' if len(variables) > 1 else '')} {set_expression}}} {self.transform(child, expression)}"

class BreakDownSumsTransformer(VisitingTransformer[ArithmeticItem]):

    def transform_sum_from_to(self, child: ArithmeticItem, variable: ArithmeticItem, from_value: ArithmeticItem, to_value: ArithmeticItem, expression: SumFromTo, _parent: ArithmeticItem) -> ArithmeticItem:
        if not isinstance(from_value, Constant):
            raise ValueError()
        if not isinstance(to_value, Constant):
            raise ValueError()

        a = from_value.value
        b = to_value.value
        new_child = self.transform(child, expression)
        initial = AssigningTransformer((variable, Constant(a))).transform(new_child)
        return functools.reduce(lambda current, next: current + AssigningTransformer((variable, Constant(next))).transform(new_child), range(a + 1, b + 1), initial)

    def transform_sum_set(self, child: ArithmeticItem, variables: list[ArithmeticItem], _set_expression: str, set_callback: Callable[[], list[ArithmeticItem | tuple[ArithmeticItem]]], expression: SumSet, _parent: ArithmeticItem) -> ArithmeticItem:
        new_child = self.transform(child, expression)
        return functools.reduce(
            lambda current, new: current + AssigningTransformer(*zip(variables, new if isinstance(new, tuple) else [new])).transform(new_child),
            set_callback(),
            0
        )

class AssigningTransformer(VisitingTransformer[ArithmeticItem]):
    assignments: list[tuple[Variable, ArithmeticItem]]

    def __init__(self, *assignments: tuple[Variable, ArithmeticItem]) -> None:
        super().__init__()
        self.assignments = list(assignments)

    def transform_variable(self, name: str, subscripts: list[ArithmeticItem], superscripts: list[ArithmeticItem], expression: Variable, parent: ArithmeticItem) -> ArithmeticItem:
        for (variable, value) in self.assignments:
            if expression == variable:
                return value
        return super().transform_variable(name, subscripts, superscripts, expression, parent)
