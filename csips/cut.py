from csips import IP, Solution

import numpy as np


def gomory(soln: Solution, prob: IP) -> IP:
    """
    Attach gomory cuts to the current problem
    """

    Aub = np.copy(prob.Aub)
    bub = np.copy(prob.bub)

    for i in range(prob.Aub.shape[0]):

        a_row = prob.Aub.getrow(i)
        b = np.dot(a_row, soln.x)

        a_row *= -1
        b *= -1

        Aub = np.vstack((Aub, a_row))
        bub = np.vstack((bub, b))

    return IP(prob.cT, Aub, bub, prob.Aeq, prob.beq, prob.bounds)


def noop(soln: Solution, prob: IP) -> IP:
    """
    A no-op cut which just returns the provided problem
    """
    return prob
