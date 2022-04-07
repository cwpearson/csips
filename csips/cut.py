from csips.problem import IP, Solution

import numpy as np

from typing import Tuple

"""
cutting functions either return a subproblem with addition constraints and true, or the original and false
"""


def gomory(soln: Solution, prob: IP) -> Tuple[IP, bool]:
    """
    Attach gomory cuts to the current problem

    This is totally borken, the constraints need to come from the simplex tableau, not A,
    which is not provided by scipy.linprog anymore
    """

    Aub = np.copy(prob.Aub)
    bub = np.copy(prob.bub)

    nrows = prob.Aub.shape[0]

    for i in range(nrows):

        a_row = prob.Aub[i, :]  # ith row
        b = np.dot(a_row, soln.x)

        # remove integral part (leaving positive fractions)
        a_row -= np.floor(a_row)
        print(f"fractional part: {a_row}")

        a_row *= -1
        b *= -1

        print(f"a vstack {Aub}, {a_row}")
        Aub = np.vstack((Aub, a_row))

        print(f"b hstack {bub}, {b}")
        bub = np.hstack((bub, b))

    return IP(prob.cT, Aub, bub, prob.Aeq, prob.beq, prob.bounds), True


def noop(soln: Solution, prob: IP) -> Tuple[IP, bool]:
    """
    A no-op cut which just returns the provided problem
    """
    return prob, False
