import pytest

import csips
import numpy as np


def test_1():
    """
    https://faculty.math.illinois.edu/~mlavrov/docs/482-spring-2020/lecture33.pdf
    max      4x+5y
    subj. to x  + 4y <= 10
             3x - 4y <= 6
             x,y     >= 0

    linear solution should be x,y = 4,1.5 with f = 23.5
    integer solution should be x,y = 2,2 with f = 18
    """

    cT = np.array([4, 5])
    cT *= -1  # convert max to min
    Aub = [
        [1, 4],
        [3, -4],
    ]
    bub = [10, 6]
    Aeq = None
    beq = None
    bounds = [
        (0, None),
        (0, None),
    ]

    ip = csips.IP(cT, Aub, bub, Aeq, beq, bounds)
    result = csips.branch_and_bound(ip)

    assert np.all(np.isclose(result[0], [2, 2]))
    assert np.all(np.isclose(result[1], -18))


def test_2():
    """
    http://www.math.clemson.edu/~mjs/courses/mthsc.440/integer.pdf
    "S 2.1"

    max 8w + 11x + 6y + 4z
    sub. to
    5w + 7x + 4y + 3z <= 14
    0 <= w,z,y,z <= 1

    LP solution is (1,1,0.5,0) -> 22
    INT is (0,1,1,1) -> 21
    """

    cT = np.array([8, 11, 6, 4])
    cT *= -1  # max
    Aub = np.array(
        [
            [5, 7, 4, 3],
        ]
    )
    bub = np.array([14])
    Aeq = None
    beq = None
    bounds = [
        (0, 1),
        (0, 1),
        (0, 1),
        (0, 1),
    ]

    ip = csips.IP(cT, Aub, bub, Aeq, beq, bounds)
    result = csips.branch_and_bound(ip)

    assert np.all(np.isclose(result[0], [0, 1, 1, 1]))
    assert np.all(np.isclose(result[1], -21))


def test_3():
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

    A few different LP solutions
    INT is (0,1,0,1) -> 15

    """

    cT = np.array([8, 11, 6, 4])
    cT *= -1  # max
    Aub = np.array(
        [
            [5, 7, 4, 3],
            [1, 1, 1, 1],
            [0, 1, 0, -1],
            [1, 0, 1, 0],
        ]
    )
    bub = np.array([14, 2, 0, 1])
    Aeq = None
    beq = None
    bounds = [
        (0, 1),
        (0, 1),
        (0, 1),
        (0, 1),
    ]

    ip = csips.IP(cT, Aub, bub, Aeq, beq, bounds)
    result = csips.branch_and_bound(ip)

    assert np.all(np.isclose(result[0], [0, 1, 0, 1]))
    assert np.all(np.isclose(result[1], -15))
