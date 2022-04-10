import numpy as np
# Type aliases
matrix = np.array
vector = np.array

eps = -1e-10 # Fractional values might not give perfect solution, so small tolerance needed

def __add_slack_vars(c: vector, A: matrix):
    num_constraint_rows = A.shape[0]
    c = np.hstack((c, np.zeros(num_constraint_rows)))
    A = np.hstack((A, np.identity(num_constraint_rows)))
    return A.shape, A, c

def __basic_feasible_solution(c: vector, A: matrix, n: int, m: int):
    B_ind = np.arange(n-m, n)
    AB = A[:, B_ind] # The slack vars start as basic feasible sol
    cB = -c[B_ind]
    N_ind = np.arange(n-m)
    AN = A[:, N_ind]
    cN = -c[N_ind]
    return AB, B_ind, AN, N_ind, cN, cB

def revised_simplex(c: vector, A: matrix, b: vector):
    (m, n), A, c = __add_slack_vars(c, A)
    AB, B_ind, AN, N_ind, cN, cB = __basic_feasible_solution(c, A, n, m)
    AB_inv = np.linalg.inv(AB) # initial inverse basis
    rc = cN - cB @ AB_inv @ AN # initial reduced costs
    b_ = AB_inv @ b # initial b vector

    while(not all(rc > eps)):
        entering_var_ind = np.argmin(rc)
        max_increase = np.inf
        leaving_var_ind = -1
        for i, value in enumerate(b_):
            if AN[i, entering_var_ind] > 0:
                increase = value / AN[i, entering_var_ind]
                if increase < max_increase:
                    max_increase = increase
                    leaving_var_ind = i
        if leaving_var_ind == -1:
            print("DEGENERACY FOUND IN TABLEAU... LOOPING!")
        else:
            # Swap columns
            AB[:, leaving_var_ind] = A[:, N_ind[entering_var_ind]]
            AN[:, entering_var_ind] = A[:, B_ind[leaving_var_ind]]

            # Swap indices
            aux = B_ind[leaving_var_ind]
            B_ind[leaving_var_ind] = N_ind[entering_var_ind]
            N_ind[entering_var_ind] = aux

            # Swap costs
            aux = cB[leaving_var_ind]
            cB[leaving_var_ind] = cN[entering_var_ind]
            cN[entering_var_ind] = aux

            # Recompute reduced cost to check for optimality
            AB_inv = np.linalg.inv(AB)
            rc = cN - cB @ AB_inv @ AN
            b_ = AB_inv @ b
    var_values = np.zeros(n)
    var_values[B_ind] = b_
    return var_values[:n-m], -cB @ b_

        


