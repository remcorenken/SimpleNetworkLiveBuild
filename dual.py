#this file will contain the definition of dual numbers
#info taken from https://blogomorphism.com/posts/dual_numbers/
from structure_decorators import *
import typing as t
import numpy as np

@subtract_using_negative
@commutative_multiplication
@commutative_addition
class DualNumber:
    def __init__(self, real: float = 0.0, dual: float = 0.0) -> None:
        self.real = real
        self.dual = dual

    def __repr__(self) -> str:
        return f"{self.real} + {self.dual}ε"

    def __add__(self, other: t.Any) -> "DualNumber":
        match other:
            case DualNumber():
                return DualNumber(
                    real=self.real + other.real, dual=self.dual + other.dual
                )
            case float() | int():
                return DualNumber(real=self.real + other, dual=self.dual)
            case _:
                return NotImplemented

    def __mul__(self, other: t.Any) -> "DualNumber":
        match other:
            case DualNumber():
                return DualNumber(
                    real=self.real * other.real,
                    dual=self.real * other.dual + other.real * self.dual,
                )
            case float() | int():
                return DualNumber(real=self.real * other, dual=self.dual * other)
            case _:
                return NotImplemented

    def __neg__(self) -> "DualNumber":
        return self.__mul__(-1)

def main():
    eps = DualNumber(dual=1)

    def dual_sin(x: DualNumber) -> DualNumber:
        return np.sin(x.real) + np.cos(x.real) * x.dual * eps

    def dual_exp(x: DualNumber) -> DualNumber:
        return np.exp(x.real) + np.exp(x.real) * x.dual * eps

    def my_func(x: DualNumber) -> DualNumber:
        return dual_exp(dual_sin(x)) * 3 + dual_sin(x) * dual_sin(x)

    def compute_derivative(f: t.Callable[[DualNumber], DualNumber], x: float) -> float:
        return f(x + eps).dual

    print(compute_derivative(my_func, 1))


if __name__ == "__main__":
    main()