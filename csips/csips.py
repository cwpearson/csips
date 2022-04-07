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
from csips import branch
from csips import cut
from csips.problem import IP, Solution

from scipy.optimize import linprog, OptimizeResult
from scipy.sparse import csr_matrix
import numpy as np

import math
import sys
from typing import NamedTuple, Any, Tuple, Union

from csips.expr import BinExpr, Var, UbCon, EqCon, ScalExpr, SumExpr, Maximize


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


def solve_lp_relaxation(ip: IP) -> Solution:
    result = linprog(
        ip.cT,
        ip.Aub,
        ip.bub,
        ip.Aeq,
        ip.beq,
        ip.bounds,
        method="revised simplex",
        options={},
    )
    print(f"result {result}")
    if not result.success:
        return Solution(None, float("inf"))
    else:
        return Solution(result.x, result.fun)


def is_integer(x, TOL):
    return abs(x - round(x)) <= TOL


def branch_and_bound(
    ip: IP,
    best=Solution(None, float("inf")),  # best solution so far
    brancher=branch.first,  # branch strategy
    cutter=cut.noop,  # cutting strategy
) -> Solution:
    """
    return (best soln, best function value) for ip

    """

    print(f"solve... (best so far is {best})")

    TOL = 1e-8

    soln = solve_lp_relaxation(ip)

    # no feasible solution
    if soln.x is None:
        print(f"no feasible LP solution")
        return best
    else:  # feasible solution, but it's not good enough
        if soln.fun >= best.fun:
            print(
                f"feasible LP solution {soln} no better than best integer solution so far {best}"
            )
            return best
        else:
            print(f"feasible LP solution {soln}")

    # if an all-integer solution, return it if it's better than the best so far,
    # otherwise return best so far
    all_integer = True
    for ix, x in enumerate(soln.x):
        if not is_integer(x, TOL):
            all_integer = False
            break
    if all_integer:
        if soln.fun < best.fun:
            print(f"integer solution {soln} is best so far (better than {best})")
            return soln
        else:
            print(f"integer solution {soln} is worse than best so far: {best}")
            return best

    # no integer solution, see if we can find some cuts
    sc, found = cutter(soln, ip)
    if found:
        print(f"added cuts, solving compared to {best}")
        return branch_and_bound(sc, best, brancher=brancher, cutter=cutter)

    # otherwise, create two subproblems for each non-integer solution
    s1, s2 = brancher(soln, ip)
    r1 = branch_and_bound(s1, best, brancher=brancher, cutter=cutter)

    if r1.fun < best.fun:
        print(f"s1 produced new best: {r1} (better than {best})")
        best = r1

    r2 = branch_and_bound(s2, best, brancher=brancher, cutter=cutter)
    if r2.fun < best.fun:
        print(f"s2 produced new best: {r2} (better than {best})")
        best = r2

    return best


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
