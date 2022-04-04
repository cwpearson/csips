"""
Solve simple integer linear programming problems
Want to support:
* Binary variables
* Integer Variables
* Linear Constraints

https://ocw.mit.edu/courses/15-053-optimization-methods-in-management-science-spring-2013/pages/lecture-notes/


1. Solve the relaxed LP
2. If some variables have fractional values, do "branch and bound"
    1. e.g. variable v = 1.5, form two subproblems, each with a new linear constraint:
        1. v <= 1
        2. v >= 2

First, we need to get the problem into the standard form


scipy.linprog
This allows us to solve a linear program

min cT * x
such that A_ub * x <= b_ub
          A_eq * x  = b_eq
                 l <=    x <= u




"""

from scipy.optimize import linprog, OptimizeResult
from scipy.sparse import csr_matrix
import numpy as np

import math
import sys
from typing import NamedTuple, Any, Tuple, Union


class IP(NamedTuple):
    """
    Inputs to a linear progrmaming solver
    """

    cT: Any
    Aub: csr_matrix
    bub: Any
    Aeq: csr_matrix
    beq: Any
    bounds: list[Tuple[Union[int, None], Union[int, None]]]


class Var:
    def __init__(self, id):
        self.id = id

    def __add__(self, rhs):
        return SumExpr([self, rhs])

    def __sub__(self, rhs):
        return SumExpr([self, ScalExpr(-1, rhs)])

    def __mul__(self, rhs):
        assert isinstance(rhs, int)
        return ScalExpr(rhs, self)

    def __rmul__(self, lhs):
        assert isinstance(lhs, int)
        return ScalExpr(lhs, self)

    def __le__(self, rhs):
        if isinstance(rhs, int):
            return UbCon(self, rhs)
        else:
            # self <= rhs
            # as
            # self - rhs <= 0
            lhs = self - rhs
            return UbCon(lhs, 0)

    def __ge__(self, rhs):
        """
        self >= rhs
        implement as
        -self <= -lhs
        """
        if isinstance(rhs, int):
            slhs = ScalExpr(-1, self)
            srhs = -1 * rhs
            return UbCon(slhs, srhs)
        else:
            slhs = ScalExpr(-1, self)
            srhs = ScalExpr(-1, rhs)
            return UbCon(slhs, srhs)

    def __repr__(self):
        return f"<{self.id}>"


class ScalExpr:
    """
    represents a * x
    a is an int
    """

    def __init__(self, a, x):
        if not isinstance(a, int):
            raise Exception(f"a was {type(a)}")
        if isinstance(x, ScalExpr):
            self.a = x.a * a
            self.x = x.x
        else:
            self.a = a
            self.x = x

    def __add__(self, rhs):
        return SumExpr([self, rhs])

    def __mul__(self, rhs):
        if isinstance(rhs, int):
            self.a *= rhs
            return self
        else:
            raise Exception(f"only multiply ScalExpr by int (got {type(rhs)})")

    def __rmul__(self, lhs):
        return self * lhs

    def __repr__(self):
        return f"{self.a} * ({self.x})"


class BinExpr:
    def __init__(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = rhs


# FIXME: remove zeros
class SumExpr:
    def __init__(self, exprs):
        self.exprs = []
        for x in exprs:
            if isinstance(x, SumExpr):  # flatten
                for y in x.exprs:
                    self.exprs.append(y)
            else:
                self.exprs.append(x)

        filtered = []
        for x in self.exprs:
            if isinstance(x, int) and x == 0:
                pass
            else:
                filtered.append(x)
        self.exprs = filtered

    def __add__(self, rhs):
        if isinstance(rhs, SumExpr):
            return SumExpr(self.exprs + rhs.exprs)
        else:
            return SumExpr(self.exprs + [rhs])

    def __radd__(self, lhs):
        if isinstance(lhs, SumExpr):
            return SumExpr(lhs.exprs + self.exprs)
        else:
            return SumExpr([lhs] + self.exprs)

    # FIXME: don't nest SumExprs
    def __sub__(self, rhs):
        return SumExpr(self.exprs + [-1 * rhs])

    # FIXME: don't nest SumExprs
    def __rsub__(self, lhs):
        return SumExpr([lhs] + [-1 * x for x in self.exprs])

    def __le__(self, rhs):
        """
        self <= rhs
        """
        if isinstance(rhs, int):
            return UbCon(self, rhs)
        else:
            raise NotImplementedError

    def __eq__(self, rhs):
        """
        self == rhs
        """
        if isinstance(rhs, int):
            return EqCon(self, rhs)
        else:
            return self + ScalExpr(-1, rhs) == 0

    def __repr__(self):
        return "(" + " + ".join(str(x) for x in self.exprs) + ")"


class UbCon(BinExpr):
    """
    lhs <= rhs
    """

    def __init__(self, lhs, rhs):
        BinExpr.__init__(self, lhs, rhs)

    def __repr__(self):
        return f"({self.lhs}) <= ({self.rhs})"


class EqCon(BinExpr):
    """
    lhs == rhs
    """

    def __init__(self, lhs, rhs):
        BinExpr.__init__(self, lhs, rhs)

    def __repr__(self):
        return f"({self.lhs}) == ({self.rhs})"


class Maximize:
    def __init__(self, expr):
        self.expr = expr

    def __repr__(self):
        return f"maximize({self.expr})"


def summation(iter):
    return SumExpr(iter)


class Model:
    def __init__(self):
        self.vid = 0
        self.vids = {}
        self.vars = {}
        self.ubs = []
        self.eqs = []
        self.obj = None

    def var_1d(self, r):
        ret = np.empty((len(r)), dtype=Var)
        for i in r:
            var = Var(self.vid)
            self.vids[var] = self.vid
            self.vars[self.vid] = var
            self.vid += 1
            ret[i] = var
        return np.array(ret)

    def var_2d(self, X, Y):
        ret = np.empty((len(X), len(Y)), dtype=Var)
        for y in Y:
            for x in X:
                var = Var(self.vid)
                self.vids[var] = self.vid
                self.vars[self.vid] = var
                self.vid += 1
                ret[x, y] = var
        return ret

    def var_3d(self, X, Y, Z):
        ret = np.empty((len(X), len(Y), len(Z)), dtype=Var)
        for z in Z:
            for y in Y:
                for x in X:
                    var = Var(self.vid)
                    self.vids[var] = self.vid
                    self.vars[self.vid] = var
                    self.vid += 1
                    ret[x, y, z] = var
        return ret

    def maximize(self, exp):
        """
        set objective to be the sum of vars
        """
        self.obj = Maximize(exp)
        return self.obj

    def constraint(self, c):
        if isinstance(c, UbCon):
            self.ubs.append(c)
        elif isinstance(c, EqCon):
            self.eqs.append(c)
        else:
            raise Exception(f"c was {type(c)}")
        return c

    def fold(e):

        if isinstance(e, ScalExpr):
            if isinstance(e.a, int) and isinstance(e.x, int):
                # int * int -> int
                return e.a * e.x
        elif isinstance(e, SumExpr):
            ints = 0
            rest = []
            for x in e.exprs:
                if isinstance(x, int):
                    ints += x
                else:
                    rest.append(Model.fold(x))
            exprs = rest + [ints]
            if len(exprs) == 1:
                return exprs[0]
            else:
                return SumExpr(exprs)

        if isinstance(e, BinExpr):
            e.lhs = Model.fold(e.lhs)
            e.rhs = Model.fold(e.rhs)
        return e

    def distribute(e):
        # print(f"distribute {e}")
        if isinstance(e, ScalExpr):
            if isinstance(e.x, SumExpr):
                # a * (x + y) -> a * x + a * y
                e = SumExpr([ScalExpr(e.a, x) for x in e.x.exprs])
                # print(f"produced sum of scals {f}")
                # return f

        if isinstance(e, SumExpr):
            # print(f"sumexpr in distribute: {e}")
            e = SumExpr([Model.distribute(x) for x in e.exprs])
            # print(f"produced sum {f}")
            # return f
        elif isinstance(e, BinExpr):
            e.lhs = Model.distribute(e.lhs)
            e.rhs = Model.distribute(e.rhs)
            # print(f"produced binary {e}")
            # return e
        # print(f"unchanged {e}")
        return e

    def move_left(e):
        if isinstance(e, UbCon):
            if isinstance(e.rhs, SumExpr):
                moved = []
                stay = []
                for x in e.rhs.exprs:
                    if not isinstance(x, int):
                        moved.append(x)
                    else:
                        stay.append(x)
                # subtract from lhs
                e.lhs = SumExpr([e.lhs] + [-1 * m for m in moved])
                e.rhs = SumExpr(stay)
                return e
            if isinstance(e.rhs, ScalExpr):
                # subtract from lhs
                e.lhs = SumExpr([e.lhs, -1 * e.rhs])
                e.rhs = 0
                return e
        return e

    def standard_form(con):
        """
        rewrite a constraint expression to be
        SumExpr <= int
        """

        con = Model.distribute(con)
        con = Model.fold(con)
        con = Model.move_left(con)
        con = Model.fold(con)

        if isinstance(con.rhs, int):
            return con
        else:
            raise Exception(f"couldn't rewrite {con}, rhs type = {type(con.rhs)}")

    def cons_coeffs(self, con):
        con = Model.standard_form(con)
        # print(f"sf: {con}")
        return self.coeffs(con.lhs), self.coeffs(con.rhs)

    def coeffs(self, e):
        if isinstance(e, SumExpr):
            row = np.zeros(self.vid)
            for x in e.exprs:
                row += self.coeffs(x)
            return row
        elif isinstance(e, ScalExpr):
            row = np.zeros(self.vid)
            if not isinstance(e.x, Var):
                raise Exception(f"e.x was {type(e.x)}")
            vid = self.vids[e.x]
            row[vid] = e.a
            return row
        elif isinstance(e, Maximize):
            row = self.coeffs(e.expr)
            row *= -1  # max as a minimization problem
            return row
        elif isinstance(e, int):
            return np.array([e])
        elif isinstance(e, Var):
            row = np.zeros(self.vid)
            vid = self.vids[e]
            row[vid] = 1
            return row
        else:
            raise Exception(f"e was {type(e)}")

    def get_ip(self) -> IP:

        print(f"ubs:  {len(self.ubs)}")
        print(f"eqs:  {len(self.eqs)}")
        print(f"vars: {len(self.vars)}")

        print(f"obj: {self.obj}")
        cT = self.coeffs(self.obj)
        # print(f"cT: {cT}")

        Aub = np.zeros((0, self.vid))
        bub = np.zeros((0, 1))
        for c in self.ubs:
            # print(f"ub: {c}")
            lhs_row, rhs_entry = self.cons_coeffs(c)
            Aub = np.vstack((Aub, lhs_row))
            bub = np.vstack((bub, rhs_entry))

        Aeq = np.zeros((0, self.vid))
        beq = np.zeros((0, 1))
        for c in self.eqs:
            # print(f"eq: {c}")
            lhs_row, rhs_entry = self.cons_coeffs(c)
            Aeq = np.vstack((Aeq, lhs_row))
            beq = np.vstack((beq, rhs_entry))

        # print(Aub)
        # print(bub)
        # print(Aeq)
        # print(beq)

        bounds = [(-1 * float("inf"), float("inf")) for _ in range(self.vid)]

        print(f"{Aub.shape[0]} rows")
        print(f"{Aub.shape[1]} cols")

        return IP(cT, Aub, bub, Aeq, beq, bounds)

    def value(self, var, x):
        """
        retrieve variable value from raw solution values
        """
        vid = self.vids[var]
        return x[vid]


def example_ip_3():
    """
    http://www.math.clemson.edu/~mjs/courses/mthsc.440/integer.pdf
    "S 2.1"

    max 8w + 11x + 6y + 4z
    sub. to
    5w + 7x + 4y + 3z <= 14
    0 <= w,z,y,z <= 1

    LP solution is (1,1,0.5,0) -> 22
    INT is (0,1,1,1) -> 21

    w+x+y+z <= 2
    x-z <= 0
    w+y <= 1
    """

    m = Model()
    x = m.var_1d(range(0, 4))
    m.constraint(5 * x[0] + x[1] * 7 + x[2] * 4 + x[3] * 3 <= 14)
    for v in x:
        m.constraint(0 <= v)
        m.constraint(v <= 1)
    m.constraint(x[0] + x[1] + x[2] + x[3] <= 2)
    m.constraint(x[1] - x[3] <= 0)
    m.constraint(x[0] + x[2] <= 1)

    m.maximize(8 * x[0] + 11 * x[1] + 6 * x[2] + 4 * x[3])

    return m.get_ip()


def solve_lp_relaxation(ip: IP) -> OptimizeResult:
    result = linprog(
        ip.cT,
        ip.Aub,
        ip.bub,
        ip.Aeq,
        ip.beq,
        ip.bounds,
        # method="highs",
        options={},
    )
    return result


def is_integer(x, TOL):
    return abs(x - round(x)) <= TOL


def branch_and_bound(
    ip: IP,
    bestX=None,  # best solution so far
    bestF=float("inf"),  # objective of best solution so far
):
    """
    return (best soln, best function value) for ip

    """

    print(f"solve... (best so far is {bestF} @ {bestX})")

    TOL = 1e-8

    result = solve_lp_relaxation(ip)
    print(f"LP Iterations: {result.nit}")

    # no feasible solution
    if not result.success:
        print(f"no feasible LP solution")
        return bestX, bestF
    else:  # feasible solution, but it's not good enough
        if result.fun >= bestF:
            print(
                f"feasible LP solution {result.fun} @ {result.x} no better than best integer solution so far {bestF} @ {bestX}"
            )
            return bestX, bestF
        else:
            print(f"feasible LP solution {result.fun} @ {result.x}")

    # if an all-integer solution, return it if it's better than the best so far,
    # otherwise return best so far
    all_integer = True
    for ix, x in enumerate(result.x):
        if not is_integer(x, TOL):
            all_integer = False
            break
    if all_integer:
        if result.fun < bestF:
            print(
                f"integer solution {result.fun} @ {result.x} is best so far (better than {bestF} @ {bestX})"
            )
            return result.x, result.fun
        else:
            print(
                f"integer solution {result.fun} @ {result.x} is worse than best so far: {bestF} @ {bestX}"
            )
            return bestX, bestF

    # otherwise, create two subproblems for each non-integer solution
    for ix, x in enumerate(result.x):
        if not is_integer(x, TOL):
            print(f"variable {ix} = {x} is not integral")
            # subproblem 1: add var[ix] <= floor(x)
            bounds = list(ip.bounds)
            lb, ub = bounds[ix][0], bounds[ix][1]

            if ub is None:
                ub = float("inf")
            bounds[ix] = lb, min(ub, math.floor(x))
            print(f"left subprob with new bounds for var {ix}: {bounds[ix]}")
            sip = IP(ip.cT, ip.Aub, ip.bub, ip.Aeq, ip.beq, bounds)
            r1x, r1f = branch_and_bound(sip, bestX, bestF)

            if r1f < bestF:
                print(
                    f"l sub produced new best: {r1f} @ {r1x} (better than {bestF} @ {bestX})"
                )
                bestF = r1f
                bestX = r1x

            # subproblem 2: add var[ix] >= ceil(x)
            bounds = list(ip.bounds)
            lb, ub = bounds[ix][0], bounds[ix][1]
            bounds[ix] = max(lb, math.ceil(x)), ub
            print(f"right subprob with new bounds for var {ix}: {bounds[ix]}")
            sip = IP(ip.cT, ip.Aub, ip.bub, ip.Aeq, ip.beq, bounds)
            r2x, r2f = branch_and_bound(sip, bestX, bestF)

            if r2f < bestF:
                print(f"r sub new best: {r2f} @ {r2x}")
                bestF = r2f
                bestX = r2x

            return bestX, bestF


# ip = example_ip_1()
# result = branch_and_bound(ip)
# print(result)

# print("")
# print("")

# ip = example_ip_2()
# result = branch_and_bound(ip)
# print(result)


# print("")
# print("")

# ip = example_ip_3()
# result = branch_and_bound(ip)
# print(result)
