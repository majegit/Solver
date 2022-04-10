from typing import Union, Tuple, Callable, Optional
import numpy as np
from .revised_simplex import revised_simplex
from .simplex import simplex
# Type aliases # TODO typehint.TypeAlias when python 3.10 comes.
matrix = np.ndarray
vector = np.ndarray

class Model:
    def __init__(self, c: list):
        self.c = np.array(c)
        self.A = []

    def __check_constraint(constraint: list) -> str:
        legal_constraint_types = {
            '<=': 'LE',
            '=<': 'LE',
            '>=': 'GE',
            '=>': 'GE',
            '=' : 'EQ',
            '==': 'EQ',
        }
        if len(constraint) < 3:
                raise ValueError(f'This constraint is invalid: {constraint}, all \
                    constraints must be lists of at least length 3. E.g. \
                    [1, 3, -2, \'<=\', 10].')
        constraint_type = constraint[-2]
        if type(constraint_type) != str:
                raise TypeError(f'The constraint: {constraint} does not have an \
                    inequality/equality \'string\' in the second to last spot. \
                    Instead it has {constraint[-2]} of type {type(constraint[-2])}.')
        normalised_constraint_type = legal_constraint_types.get(constraint_type, None)
        if normalised_constraint_type == None:
            raise ValueError(f'The constraint: {constraint} does represent an \
                inequality/equality. Instead it has: {constraint_type}. Inequalities\
                /equalities are represented by one of {legal_constraint_types.keys()}.')
        return normalised_constraint_type

    # TODO typehint return is a tuple
    def __check_method(method: Optional[str]) -> Callable[[vector, matrix, vector], float]: 
        legal_methods = {
            'simplex': simplex,
            'revised_simplex': revised_simplex,
        }
        if method == None:
            return revised_simplex
        if type(method) != str:
            raise TypeError(f'The solver method must be a string. Received {method} \
                of type {type(method)}.')
        method = legal_methods.get(method, '')
        if method == '':
            raise ValueError(f'The method: {method} is unknown, check for spelling errors. \
                Known methods are {legal_methods}.')
        return method

    def add_constraint(self, constraint: list):
        constraint_type = Model.__check_constraint(constraint)
        constraint = np.delete(constraint, -2).astype(float)
        if constraint_type != 'LE':
            self.A.append(-constraint)
        if constraint_type != 'GE':
            self.A.append(constraint)

    def solve(self, method: Optional[str] = None) -> Tuple[vector, float]:
        method = Model.__check_method(method)
        self.A = np.array(self.A)
        return method(self.c, self.A[:, :-1], self.A[:, -1])



