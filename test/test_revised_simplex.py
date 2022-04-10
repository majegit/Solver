from context import revised_simplex
import numpy as np

tol = 1e-10

def test_revised_simplex():
    c = np.array([1, 2, 3])
    A = np.array([[1, 1, 0],
                  [0, 1, 1],
                  [1, 0, 1]])
    b = np.array([20.5, 20.5, 30.5])
    var_values, obj_value = revised_simplex(c, A, b)

    expected_var_values = [10, 0, 20.5]
    assert(all(abs(var_values - expected_var_values) <= tol))

    expected_obj_value = 71.5
    assert(abs(obj_value - expected_obj_value) <= tol)
    