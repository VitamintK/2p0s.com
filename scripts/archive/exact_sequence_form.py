"""ARCHIVED: not used by the site's data pipeline.

Built to compute an exact (rational) Nash value for Leduc poker, paired
with exact_lp.py's hand-rolled exact simplex. Validated correct on Kuhn
poker (exactly reproduces -1/18), but Leduc's LP is far larger and too
degenerate for that simplex to solve in reasonable time -- see exact_lp.py
for the full story. Leduc's Nash value ended up computed approximately
instead, via cvxpy + HiGHS directly in compute_game_data.py. Kept here in
case exact solving is worth another attempt later.

Exact Nash value of a two-player zero-sum game via sequence-form LP.

The LP construction (equations (8) and (9) of Koller, Megiddo & von Stengel,
"Fast Algorithms for Finding Randomized Strategies in Game Trees", STOC '94)
is ported line-for-line from OpenSpiel's `sequence_form_lp._construct_lps`,
which is a validated reference implementation -- so the sign conventions and
tree recursion are trusted as-is. The only change is arithmetic: every
number (including chance-node probabilities) is an exact `fractions.Fraction`
rather than a float, and the final LP is solved with `exact_lp.solve_lp`
(an exact rational simplex) instead of OpenSpiel's float-based cvxpy solve.
"""

import enum
from fractions import Fraction

import pyspiel

from exact_lp import solve_lp

_EMPTY_INFOSET_KEYS = ["***EMPTY_INFOSET_P0***", "***EMPTY_INFOSET_P1***"]
_EMPTY_INFOSET_ACTION_KEYS = [
    "***EMPTY_INFOSET_ACTION_P0***", "***EMPTY_INFOSET_ACTION_P1***"
]
_DELIMITER = " -=- "


class ObjectiveType(enum.Enum):
    OBJ_MIN = 1
    OBJ_MAX = 2


class ConstraintType(enum.Enum):
    CONS_TYPE_LEQ = 1
    CONS_TYPE_GEQ = 2
    CONS_TYPE_EQ = 3


class _Variable:

    def __init__(self, vid, lb=None, ub=None):
        self.vid = vid
        self.lb = lb
        self.ub = ub


class _Constraint:

    def __init__(self, cid, ctype):
        self.cid = cid
        self.ctype = ctype
        self.coeffs = {}  # var label -> value
        self.rhs = None


class LinearProgram:
    """A minimal LP data container.

    This is a trimmed, Fraction-only copy of the bookkeeping (not the
    math) in `open_spiel.python.algorithms.LinearProgram`: just
    enough to record variables/constraints as `_construct_lps` builds them.
    Reimplemented locally to avoid that module's hard `cvxpy` import, which
    this script has no other use for (we solve with `exact_lp` instead).
    """

    def __init__(self, objective):
        self._objective = objective
        self._obj_coeffs = {}  # var label -> value
        self._vars = {}  # var label -> _Variable
        self._cons = {}  # cons label -> _Constraint
        self._var_list = []
        self._leq_cons_list = []
        self._eq_cons_list = []

    def add_or_reuse_variable(self, label, lb=None, ub=None):
        var = self._vars.get(label)
        if var is not None:
            assert var.lb == lb and var.ub == ub
            return
        var = _Variable(len(self._var_list), lb, ub)
        self._vars[label] = var
        self._var_list.append(var)

    def add_or_reuse_constraint(self, label, ctype):
        cons = self._cons.get(label)
        if cons is not None:
            assert cons.ctype == ctype
            return
        if ctype in (ConstraintType.CONS_TYPE_LEQ, ConstraintType.CONS_TYPE_GEQ):
            cons = _Constraint(len(self._leq_cons_list), ctype)
            self._cons[label] = cons
            self._leq_cons_list.append(cons)
        elif ctype == ConstraintType.CONS_TYPE_EQ:
            cons = _Constraint(len(self._eq_cons_list), ctype)
            self._cons[label] = cons
            self._eq_cons_list.append(cons)
        else:
            assert False, "Unknown constraint type"

    def set_obj_coeff(self, var_label, coeff):
        self._obj_coeffs[var_label] = coeff

    def set_cons_coeff(self, cons_label, var_label, coeff):
        self._cons[cons_label].coeffs[var_label] = coeff

    def add_to_cons_coeff(self, cons_label, var_label, add_coeff):
        val = self._cons[cons_label].coeffs.get(var_label, Fraction(0))
        self._cons[cons_label].coeffs[var_label] = val + add_coeff

    def set_cons_rhs(self, cons_label, value):
        self._cons[cons_label].rhs = value

    def get_var_id(self, label):
        return self._vars[label].vid


def _chance_prob_to_fraction(prob, max_denominator=10_000):
    """Recovers the exact rational chance probability from its float value.

    OpenSpiel's chance nodes in Kuhn/Leduc poker are simple combinatorial
    draws (e.g. 1/6, 1/5, 1/4), so the true probability always has a small
    denominator -- far below `max_denominator` -- and is exactly recoverable
    from the float via continued-fraction rounding.
    """
    return Fraction(prob).limit_denominator(max_denominator)


def _construct_lps(state, infosets, infoset_actions, infoset_action_maps,
                    chance_reach, lps, parent_is_keys, parent_isa_keys):
    """Direct Fraction port of sequence_form_lp._construct_lps (see module docstring)."""
    if state.is_terminal():
        returns = state.returns()
        payoff0 = Fraction(returns[0]) * chance_reach
        lps[0].add_or_reuse_constraint(
            parent_isa_keys[0], ConstraintType.CONS_TYPE_GEQ
        )
        lps[0].add_to_cons_coeff(parent_isa_keys[0], parent_isa_keys[1], -payoff0)
        lps[0].set_cons_coeff(parent_isa_keys[0], parent_is_keys[0], Fraction(1))
        lps[1].add_or_reuse_constraint(
            parent_isa_keys[1], ConstraintType.CONS_TYPE_LEQ
        )
        lps[1].add_to_cons_coeff(parent_isa_keys[1], parent_isa_keys[0], -payoff0)
        lps[1].set_cons_coeff(parent_isa_keys[1], parent_is_keys[1], Fraction(-1))
        return

    if state.is_chance_node():
        outcomes = state.chance_outcomes()
        fractions = [_chance_prob_to_fraction(prob) for _, prob in outcomes]
        if sum(fractions) != 1:
            raise ValueError(
                f"chance probabilities {fractions} did not reconstruct to an "
                "exact sum of 1; increase max_denominator"
            )
        for (action, _), frac in zip(outcomes, fractions):
            new_state = state.child(action)
            _construct_lps(new_state, infosets, infoset_actions, infoset_action_maps,
                            frac * chance_reach, lps, parent_is_keys, parent_isa_keys)
        return

    player = state.current_player()
    info_state = state.information_state_string(player)
    legal_actions = state.legal_actions(player)

    if player == 0:
        lps[0].add_or_reuse_variable(info_state)
        lps[0].add_or_reuse_constraint(
            parent_isa_keys[0], ConstraintType.CONS_TYPE_GEQ
        )
        lps[0].set_cons_coeff(parent_isa_keys[0], parent_is_keys[0], Fraction(1))
        lps[0].set_cons_coeff(parent_isa_keys[0], info_state, Fraction(-1))
        lps[1].add_or_reuse_constraint(
            info_state, ConstraintType.CONS_TYPE_EQ
        )
        lps[1].set_cons_coeff(info_state, parent_isa_keys[0], Fraction(-1))
    else:
        lps[1].add_or_reuse_variable(info_state)
        lps[1].add_or_reuse_constraint(
            parent_isa_keys[1], ConstraintType.CONS_TYPE_LEQ
        )
        lps[1].set_cons_coeff(parent_isa_keys[1], parent_is_keys[1], Fraction(-1))
        lps[1].set_cons_coeff(parent_isa_keys[1], info_state, Fraction(1))
        lps[0].add_or_reuse_constraint(
            info_state, ConstraintType.CONS_TYPE_EQ
        )
        lps[0].set_cons_coeff(info_state, parent_isa_keys[1], Fraction(-1))

    if info_state not in infosets[player]:
        infosets[player][info_state] = len(infosets[player])
    if info_state not in infoset_action_maps[player]:
        infoset_action_maps[player][info_state] = []

    new_parent_is_keys = parent_is_keys[:]
    new_parent_is_keys[player] = info_state

    for action in legal_actions:
        isa_key = info_state + _DELIMITER + str(action)
        if isa_key not in infoset_actions[player]:
            infoset_actions[player][isa_key] = len(infoset_actions[player])
        if isa_key not in infoset_action_maps[player][info_state]:
            infoset_action_maps[player][info_state].append(isa_key)

        if player == 0:
            lps[1].add_or_reuse_variable(isa_key, lb=0)
            lps[1].set_cons_coeff(info_state, isa_key, Fraction(1))
        else:
            lps[0].add_or_reuse_variable(isa_key, lb=0)
            lps[0].set_cons_coeff(info_state, isa_key, Fraction(1))

        new_parent_isa_keys = parent_isa_keys[:]
        new_parent_isa_keys[player] = isa_key

        new_state = state.child(action)
        _construct_lps(new_state, infosets, infoset_actions, infoset_action_maps,
                        chance_reach, lps, new_parent_is_keys, new_parent_isa_keys)


def _extract_lp(lp):
    """Reads a solved-form `LinearProgram` into `exact_lp.solve_lp` inputs.

    Reaches into `LinearProgram`'s private attributes because its public API
    only exposes a float/cvxpy `solve()`; the attributes themselves are a
    stable, simple data structure (variable/constraint lists), not solver
    logic, so reading them directly is safe.
    """
    num_vars = len(lp._var_list)  # pylint: disable=protected-access
    free_vars = set()
    for label, var in lp._vars.items():  # pylint: disable=protected-access
        assert var.ub is None, f"unexpected upper bound on {label!r}"
        assert var.lb is None or var.lb == 0, f"unexpected lower bound on {label!r}"
        if var.lb is None:
            free_vars.add(var.vid)

    obj = {}
    for label, coeff in lp._obj_coeffs.items():  # pylint: disable=protected-access
        obj[lp._vars[label].vid] = Fraction(coeff)  # pylint: disable=protected-access

    op_by_ctype = {
        ConstraintType.CONS_TYPE_LEQ: "<=",
        ConstraintType.CONS_TYPE_GEQ: ">=",
        ConstraintType.CONS_TYPE_EQ: "=",
    }
    constraints = []
    for cons in lp._leq_cons_list + lp._eq_cons_list:  # pylint: disable=protected-access
        coeffs = {
            lp._vars[var_label].vid: Fraction(value)  # pylint: disable=protected-access
            for var_label, value in cons.coeffs.items()
        }
        rhs = cons.rhs if cons.rhs is not None else Fraction(0)
        constraints.append((coeffs, op_by_ctype[cons.ctype], Fraction(rhs)))

    sense = "min" if lp._objective == ObjectiveType.OBJ_MIN else "max"  # pylint: disable=protected-access
    return num_vars, obj, constraints, sense, free_vars


def solve_zero_sum_game(game):
    """Solves a two-player zero-sum game exactly via sequence-form LPs.

    Args:
      game: the spiel game to solve (must be zero-sum, sequential, and have
        chance mode of deterministic or explicit stochastic).

    Returns:
      A (player 0 value, player 1 value) tuple of exact Fractions.
    """
    assert game.num_players() == 2
    assert game.get_type().utility == pyspiel.GameType.Utility.ZERO_SUM
    assert game.get_type().dynamics == pyspiel.GameType.Dynamics.SEQUENTIAL
    assert (
        game.get_type().chance_mode == pyspiel.GameType.ChanceMode.DETERMINISTIC
        or game.get_type().chance_mode
        == pyspiel.GameType.ChanceMode.EXPLICIT_STOCHASTIC
    )

    infosets = [{_EMPTY_INFOSET_KEYS[0]: 0}, {_EMPTY_INFOSET_KEYS[1]: 0}]
    infoset_actions = [{
        _EMPTY_INFOSET_ACTION_KEYS[0]: 0
    }, {
        _EMPTY_INFOSET_ACTION_KEYS[1]: 0
    }]
    infoset_action_maps = [{}, {}]
    lps = [
        LinearProgram(ObjectiveType.OBJ_MIN),  # Eq. (8)
        LinearProgram(ObjectiveType.OBJ_MAX),  # Eq. (9)
    ]
    lps[0].add_or_reuse_variable(_EMPTY_INFOSET_ACTION_KEYS[1], lb=0)  # y root
    lps[0].add_or_reuse_variable(_EMPTY_INFOSET_KEYS[0])  # p root
    lps[1].add_or_reuse_variable(_EMPTY_INFOSET_ACTION_KEYS[0], lb=0)  # x root
    lps[1].add_or_reuse_variable(_EMPTY_INFOSET_KEYS[1])  # q root
    lps[0].set_obj_coeff(_EMPTY_INFOSET_KEYS[0], Fraction(1))  # e^t p
    lps[1].set_obj_coeff(_EMPTY_INFOSET_KEYS[1], Fraction(-1))  # -q^t f
    lps[0].add_or_reuse_constraint(
        _EMPTY_INFOSET_KEYS[1], ConstraintType.CONS_TYPE_EQ
    )
    lps[0].set_cons_coeff(_EMPTY_INFOSET_KEYS[1], _EMPTY_INFOSET_ACTION_KEYS[1],
                           Fraction(-1))
    lps[0].set_cons_rhs(_EMPTY_INFOSET_KEYS[1], Fraction(-1))
    lps[1].add_or_reuse_constraint(
        _EMPTY_INFOSET_KEYS[0], ConstraintType.CONS_TYPE_EQ
    )
    lps[1].set_cons_coeff(_EMPTY_INFOSET_KEYS[0], _EMPTY_INFOSET_ACTION_KEYS[0],
                           Fraction(1))
    lps[1].set_cons_rhs(_EMPTY_INFOSET_KEYS[0], Fraction(1))

    _construct_lps(game.new_initial_state(), infosets, infoset_actions,
                    infoset_action_maps, Fraction(1), lps, _EMPTY_INFOSET_KEYS[:],
                    _EMPTY_INFOSET_ACTION_KEYS[:])

    values = []
    for lp in lps:
        num_vars, obj, constraints, sense, free_vars = _extract_lp(lp)
        result = solve_lp(num_vars, obj, constraints, sense=sense, free_vars=free_vars)
        assert result.status == "optimal", "sequence-form LP was infeasible"
        values.append(result.x)

    value0 = values[0][lps[0].get_var_id(_EMPTY_INFOSET_KEYS[0])]
    value1 = values[1][lps[1].get_var_id(_EMPTY_INFOSET_KEYS[1])]
    assert value0 == -value1, f"primal/dual mismatch: {value0} != {-value1}"
    return value0, value1


def nash_value(game_name):
    """Returns the exact Nash value to player 0 of `game_name`, as a Fraction."""
    game = pyspiel.load_game(game_name)
    value0, _ = solve_zero_sum_game(game)
    return value0
