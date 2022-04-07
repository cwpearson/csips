from scipy.sparse import csr_matrix

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


class Solution(NamedTuple):
    x: Any
    fun: float
