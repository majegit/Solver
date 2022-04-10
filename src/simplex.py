import numpy as np
# Type aliases
matrix = np.ndarray
vector = np.ndarray

eps = -1e-10 # Fractional values might not give perfect solution, so small tolerance needed

def __add_slack_vars(c: vector, A: matrix):
    num_constraint_rows = A.shape[0]
    c = np.hstack((c, np.zeros(num_constraint_rows)))
    A = np.hstack((A, np.identity(num_constraint_rows)))
    return A.shape, A

def __basic_feasible_solution(c: vector, n: int, m: int):
    B = np.arange(n-m, n) # The slack vars start as basic feasible sol
    rc = np.hstack((-c, np.zeros(m)))
    return B, rc

def simplex(c: vector, A: matrix, b: vector):
    """Takes a cost vector and a set of constraints specified by A<=b.
    """
    (m, n), A = __add_slack_vars(c, A)
    B, rc = __basic_feasible_solution(c, n, m)
    A = np.vstack((A, rc))
    b = np.hstack((b, [0]))
    A = np.hstack((A, b[:, np.newaxis]))
    del c, b, rc

    loop_counter = 0
    while(not all(A[-1, :-1] > eps)): # not optimal sol
        min_rc_col = np.argmin(A[-1, :-1])
        max_variable_inc = np.inf
        max_variable_inc_row = -1
        for row in range(m): # Go through each row and find the possible increase
            ai = A[row, min_rc_col]
            if ai > 0:
                variable_increase = A[row, -1] / ai
                if variable_increase < max_variable_inc:
                    max_variable_inc_row = row
                    max_variable_inc = variable_increase

        if(max_variable_inc_row != -1): # Not looping
            loop_counter = 0
            B[max_variable_inc_row] = min_rc_col
            A[max_variable_inc_row] *= 1/A[max_variable_inc_row, min_rc_col]
            for row in range(m+1):
                if(row != max_variable_inc_row):
                    A[row] -= A[row, min_rc_col] * A[max_variable_inc_row]

        else:
            loop_counter += 1
            if(loop_counter == 100):
                print("Looped 100 times.")

    variable_values = [0] * n
    for row, variable_index in enumerate(B):
        variable_values[variable_index] = A[row, -1]
    return np.array(variable_values[:n-m]), A[-1, -1]
        





