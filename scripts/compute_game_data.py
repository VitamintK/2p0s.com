import json
from pathlib import Path

import cvxpy as cp
import pyspiel
from open_spiel.python.algorithms import lp_solver
from open_spiel.python.algorithms.expected_game_score import policy_value
from open_spiel.python.algorithms.exploitability import nash_conv
from open_spiel.python.algorithms.get_all_states import get_all_states
from open_spiel.python.algorithms.sequence_form_lp import (
    _EMPTY_INFOSET_ACTION_KEYS,
    _EMPTY_INFOSET_KEYS,
    _construct_lps,
)
from open_spiel.python.policy import UniformRandomPolicy

DATA_DIR = Path(__file__).parent.parent / "data" / "games"


def compute_nash_value(game) -> float:
    """Solves the sequence-form LP for the game's Nash equilibrium value (to player 0).

    This is an approximate (floating-point) solve via HiGHS -- fine for
    games too large for an exact rational solve. OpenSpiel's own
    `sequence_form_lp.solve_zero_sum_game` hardcodes solver_kwargs tuned for
    ECOS (not installed here), so the LP is built and solved directly.
    """
    infosets = [{_EMPTY_INFOSET_KEYS[0]: 0}, {_EMPTY_INFOSET_KEYS[1]: 0}]
    infoset_actions = [{_EMPTY_INFOSET_ACTION_KEYS[0]: 0}, {_EMPTY_INFOSET_ACTION_KEYS[1]: 0}]
    infoset_action_maps = [{}, {}]
    lps = [
        lp_solver.LinearProgram(lp_solver.ObjectiveType.OBJ_MIN),  # Eq. (8)
        lp_solver.LinearProgram(lp_solver.ObjectiveType.OBJ_MAX),  # Eq. (9)
    ]
    lps[0].add_or_reuse_variable(_EMPTY_INFOSET_ACTION_KEYS[1], lb=0)  # y root
    lps[0].add_or_reuse_variable(_EMPTY_INFOSET_KEYS[0])  # p root
    lps[1].add_or_reuse_variable(_EMPTY_INFOSET_ACTION_KEYS[0], lb=0)  # x root
    lps[1].add_or_reuse_variable(_EMPTY_INFOSET_KEYS[1])  # q root
    lps[0].set_obj_coeff(_EMPTY_INFOSET_KEYS[0], 1.0)
    lps[1].set_obj_coeff(_EMPTY_INFOSET_KEYS[1], -1.0)
    lps[0].add_or_reuse_constraint(_EMPTY_INFOSET_KEYS[1], lp_solver.ConstraintType.CONS_TYPE_EQ)
    lps[0].set_cons_coeff(_EMPTY_INFOSET_KEYS[1], _EMPTY_INFOSET_ACTION_KEYS[1], -1.0)
    lps[0].set_cons_rhs(_EMPTY_INFOSET_KEYS[1], -1.0)
    lps[1].add_or_reuse_constraint(_EMPTY_INFOSET_KEYS[0], lp_solver.ConstraintType.CONS_TYPE_EQ)
    lps[1].set_cons_coeff(_EMPTY_INFOSET_KEYS[0], _EMPTY_INFOSET_ACTION_KEYS[0], 1.0)
    lps[1].set_cons_rhs(_EMPTY_INFOSET_KEYS[0], 1.0)
    _construct_lps(
        game.new_initial_state(), infosets, infoset_actions, infoset_action_maps,
        1.0, lps, _EMPTY_INFOSET_KEYS[:], _EMPTY_INFOSET_ACTION_KEYS[:],
    )

    sol0 = lps[0].solve(solver=cp.HIGHS, solver_kwargs={})
    sol1 = lps[1].solve(solver=cp.HIGHS, solver_kwargs={})
    value0 = sol0[lps[0].get_var_id(_EMPTY_INFOSET_KEYS[0])]
    value1 = sol1[lps[1].get_var_id(_EMPTY_INFOSET_KEYS[1])]
    assert abs(value0 + value1) < 1e-6, f"primal/dual mismatch: {value0} vs {-value1}"
    return value0


def compute_and_save(spiel_name: str, slug: str) -> None:
    game = pyspiel.load_game(spiel_name)

    json_path = DATA_DIR / f"{slug}.json"
    data = json.loads(json_path.read_text())

    # --- Predefined game properties (directly from OpenSpiel) ---
    data["payoff_range"] = {
        "min": game.min_utility(),
        "max": game.max_utility(),
    }

    # --- General game structure ---
    data["num_histories"] = {
        "all": len(get_all_states(game, include_terminals=True, include_chance_states=True)),
        "decision_terminal": len(get_all_states(game, include_terminals=True, include_chance_states=False)),
        "decision": len(get_all_states(game, include_terminals=False, include_chance_states=False)),
    }

    # --- Nash equilibrium value (skip if already set, e.g. a known exact
    # fraction entered by hand) ---
    if data.get("nash_value") is None:
        data["nash_value"] = compute_nash_value(game)

    # --- Uniform random policy ---
    uniform = UniformRandomPolicy(game)
    ev = policy_value(game.new_initial_state(), [uniform, uniform])
    nc = nash_conv(game, uniform, return_only_nash_conv=False)
    # player_improvements[i] = how much player i gains by deviating to best response
    ev_p0_vs_br = ev[0] - nc.player_improvements[1]  # p0 uniform, p1 exploits
    ev_p1_vs_br = -ev[0] - nc.player_improvements[0]  # p1 uniform, p0 exploits
    data["uniform_random"] = {
        "ev_p0": ev[0],
        "ev_p1": ev[1],
        "exploitability": nc.nash_conv / 2,
        "ev_p0_vs_br": ev_p0_vs_br,
        "ev_p1_vs_br": ev_p1_vs_br,
    }

    json_path.write_text(json.dumps(data, indent=2) + "\n")

    print(f"{data['name']}:")
    print(f"  Nash value:     {data['nash_value']}")
    print(f"  EV:             P0={ev[0]:+.6f}, P1={ev[1]:+.6f}")
    print(f"  Exploitability: {nc.nash_conv / 2:.6f}")
    print(f"  P0 uniform vs BR(P1): {ev_p0_vs_br:+.6f}")
    print(f"  P1 uniform vs BR(P0): {ev_p1_vs_br:+.6f}")


if __name__ == "__main__":
    compute_and_save("kuhn_poker", "kuhn-poker")
    compute_and_save("leduc_poker", "leduc-poker")
    compute_and_save("liars_dice", "liars-dice")
