from fractions import Fraction

from exact_lp import solve_lp


def test_maximize_with_inequality_constraints():
    # max x + y  s.t.  3x + 2y <= 7,  x + 4y <= 6,  x,y >= 0
    # Exact optimum: x = 8/5, y = 11/10, objective = 27/10.
    result = solve_lp(
        num_vars=2,
        obj={0: Fraction(1), 1: Fraction(1)},
        constraints=[
            ({0: Fraction(3), 1: Fraction(2)}, "<=", Fraction(7)),
            ({0: Fraction(1), 1: Fraction(4)}, "<=", Fraction(6)),
        ],
        sense="max",
    )
    assert result.status == "optimal"
    assert result.objective == Fraction(27, 10)
    assert result.x[0] == Fraction(8, 5)
    assert result.x[1] == Fraction(11, 10)


def test_free_variable_bound_by_equality_and_inequality():
    # minimize p  s.t.  p - y >= 0,  y = 1,  y >= 0,  p free.
    # p is only bounded below (by y), so the minimum pins p == y == 1.
    result = solve_lp(
        num_vars=2,
        obj={0: Fraction(1)},
        constraints=[
            ({0: Fraction(1), 1: Fraction(-1)}, ">=", Fraction(0)),
            ({1: Fraction(1)}, "=", Fraction(1)),
        ],
        sense="min",
        free_vars={0},
    )
    assert result.status == "optimal"
    assert result.objective == Fraction(1)
    assert result.x[0] == Fraction(1)
    assert result.x[1] == Fraction(1)


def test_infeasible_constraints_are_reported():
    # x >= 0 and x <= -1 can never both hold.
    result = solve_lp(
        num_vars=1,
        obj={0: Fraction(1)},
        constraints=[
            ({0: Fraction(1)}, "<=", Fraction(-1)),
        ],
        sense="min",
    )
    assert result.status == "infeasible"
