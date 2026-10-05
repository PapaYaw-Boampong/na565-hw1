import numpy as np

def ACC_Controller(t, x, param):
    vd = param["vd"]
    v0 = param["v0"]
    m = param["m"]
    Cag = param["Cag"]
    Cdg = param["Cdg"]

    # cost function and constraints for the QP
    P = np.zeros((2,2))
    q = np.zeros([2, 1])
    A = np.zeros([5, 2])
    b = np.zeros([5])
    
    #############################################################################
    #                    TODO: Implement your code here                         #
    #############################################################################

    D = x[0]
    v = x[1]

    # # set the parameters
    lam = 5.0
    alpha = 1.5
    w = 10.0

 

    # speed goal h and safety meter B:
    # (braking term only counts when the ego car is faster than the lead)
    h = 0.5 * (v - vd) ** 2
    dv = max(v - v0, 0.0)
    B = D - 1.8 * v - 0.5 * dv ** 2 / Cdg

    # construct the cost function
    P = np.array(
        [
            [2.0, 0.0],
            [0.0, 2.0*w]
        ]
    )
    q = np.zeros([2,1])
    
    # construct the constraints
    # delta here is m times the notebook's delta, which w absorbs)
    A = np.array([
        [v - vd,                  -1.0],   # speed goal:  (v-vd)Fw - delta <= -m*lam*h
        [1.8 + dv / Cdg,           0.0],   # safety rule: -dB/dt <= alpha*B, times m
        [1.0,                      0.0],   # Fw <=  m*Cag
        [-1.0,                     0.0],   # Fw >= -m*Cdg
        [0.0,                     -1.0],   # delta >= 0
    ])
    
    b = np.array([
        -m * lam * h,
        m * (v0 - v) + m * alpha * B,
        m * Cag,
        m * Cdg,
        0.0,
    ])

    #############################################################################
    #                            END OF YOUR CODE                               #
    #############################################################################
    
    return A, b, P, q