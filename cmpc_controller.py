import numpy as np
import cvxpy as cp

def calc_Jacobian(x, u, param):

    L_f = param["L_f"]
    L_r = param["L_r"]
    dt   = param["h"]

    psi = x[2]
    v   = x[3]
    delta = u[1]
    a   = u[0]

    # Jacobian of the system dynamics
    A = np.zeros((4, 4))
    B = np.zeros((4, 2))

    #############################################################################
    #                    TODO: Implement your code here                         #
    #############################################################################

  
    beta = np.arctan(L_r / (L_r + L_f) * np.arctan(delta))
    z = L_r * np.arctan(delta) / (L_f + L_r)
    # d(beta)/d(delta), chain rule through both arctans
    dbeta = (L_r / (L_f + L_r)) / ((1 + delta ** 2) * (1 + z ** 2))

    A = np.eye(4)
    A[0, 2] = -dt * v * np.sin(psi + beta)
    A[0, 3] =  dt * np.cos(psi + beta)
    A[1, 2] =  dt * v * np.cos(psi + beta)
    A[1, 3] =  dt * np.sin(psi + beta)
    A[2, 3] =  dt * np.sin(beta) / L_r

    B[0, 1] = -dt * v * np.sin(psi + beta) * dbeta
    B[1, 1] =  dt * v * np.cos(psi + beta) * dbeta
    B[2, 1] =  dt * v / L_r * np.cos(beta) * dbeta
    B[3, 0] =  dt

    #############################################################################
    #                            END OF YOUR CODE                               #
    #############################################################################

    return [A, B]

def LQR_Controller(x_bar, u_bar, x0, param):
    len_state = x_bar.shape[0]
    len_ctrl  = u_bar.shape[0]
    dim_state = x_bar.shape[1]
    dim_ctrl  = u_bar.shape[1]

    n_u = len_ctrl * dim_ctrl
    n_x = len_state * dim_state
    n_var = n_u + n_x

    n_eq  = dim_state * len_ctrl # dynamics
    n_ieq = dim_ctrl * len_ctrl  # input constraints

    
    #############################################################################
    #                    TODO: Implement your code here                         #
    #############################################################################

    # define the parameters
    Q = np.eye(4)  * 1.
    R = np.diag([0.1, 5.])   # steering weighted more: keeps steering where the linear model holds
    Pt = np.eye(4) * 10.

    # define the cost function
    # layout: x = [ds_0, ..., ds_N, du_0, ..., du_{N-1}]
    # block-diagonal: Q on ds_0..ds_{N-1}, Pt on ds_N, R on every du_k
    P = np.zeros((n_var, n_var))
    for k in range(len_ctrl):
        i = k * dim_state
        P[i:i + dim_state, i:i + dim_state] = Q
        j = n_x + k * dim_ctrl
        P[j:j + dim_ctrl, j:j + dim_ctrl] = R
    i = len_ctrl * dim_state
    P[i:i + dim_state, i:i + dim_state] = Pt
    q = np.zeros(n_var)

    # define the constraints
    # rows 0..n_eq-1:  A_k ds_k + B_k du_k - ds_{k+1} = 0
    # last dim_state rows: ds_0 = x0 - x_bar[0]
    A = np.zeros((n_eq + dim_state, n_var))
    b = np.zeros(n_eq + dim_state)
    for k in range(len_ctrl):
        A_k, B_k = calc_Jacobian(x_bar[k, :], u_bar[k, :], param)
        r = k * dim_state
        A[r:r + dim_state, r:r + dim_state] = A_k
        A[r:r + dim_state, r + dim_state:r + 2 * dim_state] = -np.eye(dim_state)
        j = n_x + k * dim_ctrl
        A[r:r + dim_state, j:j + dim_ctrl] = B_k
    A[n_eq:, :dim_state] = np.eye(dim_state)
    b[n_eq:] = x0 - x_bar[0, :]

    # Define and solve the CVXPY problem.
    x = cp.Variable(n_var)
    prob = cp.Problem(cp.Minimize(0.5 * cp.quad_form(x, P) + q @ x),
                      [A @ x == b])
    prob.solve(verbose=False, max_iter = 10000)


    #############################################################################
    #                            END OF YOUR CODE                               #
    #############################################################################

    u_act = x.value[n_x:n_x + dim_ctrl] + u_bar[0, :]
    return u_act

def CMPC_Controller(x_bar, u_bar, x0, param):
    len_state = x_bar.shape[0]
    len_ctrl  = u_bar.shape[0]
    dim_state = x_bar.shape[1]
    dim_ctrl  = u_bar.shape[1]
    
    n_u = len_ctrl * dim_ctrl
    n_x = len_state * dim_state
    n_var = n_u + n_x

    n_eq  = dim_state * len_ctrl # dynamics
    n_ieq = dim_ctrl * len_ctrl # input constraints

    a_limit = param["a_lim"]
    delta_limit = param["delta_lim"]
    
    #############################################################################
    #                    TODO: Implement your code here                         #
    #############################################################################
    
    # define the parameters
    Q = np.eye(4)  * 1.
    R = np.diag([0.1, 10.])  # steering weighted more: keeps steering where the linear model holds
    Pt = np.eye(4) * 10.

    # define the cost function
    # block-diagonal: Q on ds_0..ds_{N-1}, Pt on ds_N, R on every du_k
    P = np.zeros((n_var, n_var))
    for k in range(len_ctrl):
        i = k * dim_state
        P[i:i + dim_state, i:i + dim_state] = Q
        j = n_x + k * dim_ctrl
        P[j:j + dim_ctrl, j:j + dim_ctrl] = R
    i = len_ctrl * dim_state
    P[i:i + dim_state, i:i + dim_state] = Pt
    q = np.zeros(n_var)

    # define the constraints
    # rows 0..n_eq-1:  A_k ds_k + B_k du_k - ds_{k+1} = 0
    # last dim_state rows: ds_0 = x0 - x_bar[0]
    A = np.zeros((n_eq + dim_state, n_var))
    b = np.zeros(n_eq + dim_state)
    for k in range(len_ctrl):
        A_k, B_k = calc_Jacobian(x_bar[k, :], u_bar[k, :], param)
        r = k * dim_state
        A[r:r + dim_state, r:r + dim_state] = A_k
        A[r:r + dim_state, r + dim_state:r + 2 * dim_state] = -np.eye(dim_state)
        j = n_x + k * dim_ctrl
        A[r:r + dim_state, j:j + dim_ctrl] = B_k
    A[n_eq:, :dim_state] = np.eye(dim_state)
    b[n_eq:] = x0 - x_bar[0, :]

    # input limits on the real input u = u_bar + du, so shift the bounds by u_bar
    G = np.zeros((n_ieq, n_var))
    G[:, n_x:] = np.eye(n_u)
    u_min = np.array([a_limit[0], delta_limit[0]])
    u_max = np.array([a_limit[1], delta_limit[1]])
    lb = (u_min - u_bar).reshape(-1)
    ub = (u_max - u_bar).reshape(-1)

    # Define and solve the CVXPY problem.
    x = cp.Variable(n_var)
    prob = cp.Problem(cp.Minimize(0.5 * cp.quad_form(x, P) + q @ x),
                      [A @ x == b, G @ x <= ub, G @ x >= lb])
    prob.solve(verbose=False, max_iter = 10000)

    #############################################################################
    #                            END OF YOUR CODE                               #
    #############################################################################
    
    u_act = x.value[n_x:n_x + dim_ctrl] + u_bar[0, :]
    return u_act