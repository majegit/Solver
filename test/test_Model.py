from context import Model
import numpy as np
import pytest

tol = 1e-10

@pytest.mark.parametrize("method", [None, 'simplex', 'revised_simplex'])
def test_Model(method):
    c = np.array([1, 2, 3])
    m = Model(c)
    m.add_constraint([1, 1, 0, '<=', 20.5])
    m.add_constraint([0, 1, 1, '<=', 20.5])
    m.add_constraint([1, 0, 1, '<=', 30.5])
    var_values, obj_value = m.solve(method=method)

    expected_var_values = [10, 0, 20.5]
    assert(all(abs(var_values - expected_var_values) <= tol))

    expected_obj_value = 71.5
    assert(abs(obj_value - expected_obj_value) <= tol)