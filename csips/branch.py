from csips.problem import IP, Solution

import numpy as np

from typing import Tuple
import math


def is_integer(x, tol=1e-8):
    return abs(x - round(x)) <= tol


def first(soln: Solution, prob: IP) -> Tuple[IP, IP]:
    """
    branch on the first noninteger variable in the solution
    """
    for ix, x in enumerate(soln.x):
        if not is_integer(x):
            print(f"variable {ix} = {x} is not integral")
            # subproblem 1: add var[ix] <= floor(x)
            bounds = list(prob.bounds)
            lb, ub = bounds[ix][0], bounds[ix][1]

            if ub is None:
                ub = float("inf")
            bounds[ix] = lb, min(ub, math.floor(x))
            print(f"left subprob with new bounds for var {ix}: {bounds[ix]}")
            s1 = IP(prob.cT, prob.Aub, prob.bub, prob.Aeq, prob.beq, bounds)

            # subproblem 2: add var[ix] >= ceil(x)
            bounds = list(prob.bounds)
            lb, ub = bounds[ix][0], bounds[ix][1]
            bounds[ix] = max(lb, math.ceil(x)), ub
            print(f"right subprob with new bounds for var {ix}: {bounds[ix]}")
            s2 = IP(prob.cT, prob.Aub, prob.bub, prob.Aeq, prob.beq, bounds)

            return s1, s2

    raise Exception("counldn't branch on integer solution")


def most_infeasible(soln: Solution, prob: IP) -> Tuple[IP, IP]:
    """
    branch on the variable that is the "least integerish", i.e. the variable with the max of abs(x - round(x))
    """

    dist = np.abs(soln.x - np.round(soln.x))
    ix = np.argmax(dist)
    x = soln.x[ix]

    if not is_integer(x):
        print(f"variable {ix} = {x} is most infeasible")
        # subproblem 1: add var[ix] <= floor(x)
        bounds = list(prob.bounds)
        lb, ub = bounds[ix][0], bounds[ix][1]

        if ub is None:
            ub = float("inf")
        bounds[ix] = lb, min(ub, math.floor(x))
        print(f"left subprob with new bounds for var {ix}: {bounds[ix]}")
        s1 = IP(prob.cT, prob.Aub, prob.bub, prob.Aeq, prob.beq, bounds)

        # subproblem 2: add var[ix] >= ceil(x)
        bounds = list(prob.bounds)
        lb, ub = bounds[ix][0], bounds[ix][1]
        bounds[ix] = max(lb, math.ceil(x)), ub
        print(f"right subprob with new bounds for var {ix}: {bounds[ix]}")
        s2 = IP(prob.cT, prob.Aub, prob.bub, prob.Aeq, prob.beq, bounds)

        return s1, s2

    raise Exception("counldn't branch on integer solution")
