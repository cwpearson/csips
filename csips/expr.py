
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