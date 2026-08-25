"""ARCHIVED: not used by the site's data pipeline.

Built while trying to compute an exact (rational) Nash value for Leduc
poker via sequence-form LP, paired with exact_sequence_form.py. Validated
correct on Kuhn poker (exactly reproduces -1/18), but Leduc's LP
(~1,560 variables/constraints per LP) is severely degenerate, and this
hand-rolled dense-tableau simplex couldn't solve it in reasonable time --
both plain Bland's rule and a Dantzig's-rule-with-Bland's-fallback variant
stalled for many minutes without converging. Leduc's Nash value ended up
computed approximately instead, via cvxpy + HiGHS directly in
compute_game_data.py. Kept here in case exact solving is worth another
attempt later (e.g. a sparse revised simplex or an external exact-rational
solver).

Exact-arithmetic linear programming via a two-phase primal simplex.

Every coefficient is a `fractions.Fraction`, and every pivot is exact
Fraction arithmetic, so the reported optimum is exact -- not a
floating-point approximation. Bland's rule is used for column/row
selection, which guarantees termination even on degenerate LPs.
"""

from dataclasses import dataclass
from fractions import Fraction


@dataclass
class LPResult:
    status: str  # "optimal" or "infeasible"
    x: list  # solution values for the original variables, as Fractions
    objective: Fraction | None


def solve_lp(num_vars, obj, constraints, sense="min", free_vars=frozenset(), progress=None):
    """Solves a linear program exactly.

    Args:
      num_vars: number of original decision variables.
      obj: dict mapping variable index -> Fraction objective coefficient.
      constraints: list of (coeffs, op, rhs), where coeffs is a dict mapping
        variable index -> Fraction coefficient, op is one of "<=", ">=", "=",
        and rhs is a Fraction.
      sense: "min" or "max".
      free_vars: set of variable indices that are unrestricted in sign
        (default: all variables are constrained to be >= 0).

    Returns:
      An LPResult. If infeasible, `x` and `objective` are None.
    """
    sign = Fraction(1) if sense == "min" else Fraction(-1)

    # Split each free variable x_j into x_j = xp_j - xm_j, both >= 0.
    var_pos = {}
    var_neg = {}
    columns = []  # list of (kind, orig_index) for every simplex column
    for j in range(num_vars):
        var_pos[j] = len(columns)
        columns.append(("pos", j))
        if j in free_vars:
            var_neg[j] = len(columns)
            columns.append(("neg", j))

    def expand(coeffs):
        row = {}
        for j, value in coeffs.items():
            row[var_pos[j]] = row.get(var_pos[j], Fraction(0)) + value
            if j in var_neg:
                row[var_neg[j]] = row.get(var_neg[j], Fraction(0)) - value
        return row

    n_structural = len(columns)
    obj_row = expand({j: sign * c for j, c in obj.items()})

    # Add one slack/surplus column per inequality; equalities get none.
    # A "<=" row that ends up with rhs >= 0 keeps its lone +1 slack column,
    # which is already a valid identity basic column -- so it needs no
    # artificial variable in phase 1, unlike ">=" (surplus is -1) or "="
    # rows.
    rows = []
    for coeffs, op, rhs in constraints:
        row = expand(coeffs)
        rhs = Fraction(rhs)
        usable_slack = None
        if op == "<=":
            slack_col = len(columns)
            columns.append(("slack", None))
            row[slack_col] = Fraction(1)
            usable_slack = slack_col
        elif op == ">=":
            slack_col = len(columns)
            columns.append(("slack", None))
            row[slack_col] = Fraction(-1)
        elif op == "=":
            pass
        else:
            raise ValueError(f"unknown constraint operator {op!r}")
        if rhs < 0:
            row = {c: -v for c, v in row.items()}
            rhs = -rhs
            usable_slack = None  # the flipped slack coefficient is now -1
        rows.append((row, rhs, usable_slack))

    n_total = len(columns)
    m = len(rows)

    # Phase 1: rows with a usable slack start basic on that slack; every
    # other row gets an artificial variable, and phase 1 minimizes their sum.
    artificial_cols = []
    tableau = []
    basis = []
    for row, rhs, usable_slack in rows:
        if usable_slack is not None:
            tableau.append((dict(row), rhs))
            basis.append(usable_slack)
        else:
            art_col = n_total + len(artificial_cols)
            artificial_cols.append(art_col)
            full_row = dict(row)
            full_row[art_col] = Fraction(1)
            tableau.append((full_row, rhs))
            basis.append(art_col)

    n_with_art = n_total + len(artificial_cols)
    phase1_obj = {c: Fraction(1) for c in artificial_cols}

    status = _run_simplex(tableau, basis, phase1_obj, n_with_art, progress=progress)
    assert status == "optimal"
    phase1_value = _objective_value(tableau, basis, phase1_obj)
    if phase1_value != 0:
        return LPResult(status="infeasible", x=None, objective=None)

    # Drive any remaining (zero-valued) artificial variables out of the
    # basis before dropping the artificial columns.
    for i, b in enumerate(basis):
        if b in artificial_cols:
            row, rhs = tableau[i]
            replaced = False
            for c in range(n_total):
                if c not in basis and row.get(c, Fraction(0)) != 0:
                    tableau[i] = _pivot_row(row, rhs, c)
                    basis[i] = c
                    _eliminate_column(tableau, i, c)
                    replaced = True
                    break
            if not replaced:
                # Row is a redundant all-zero constraint; leave it be.
                assert rhs == 0

    tableau = [
        ({c: v for c, v in row.items() if c < n_total}, rhs)
        for row, rhs in tableau
    ]

    status = _run_simplex(tableau, basis, obj_row, n_total, progress=progress)
    assert status == "optimal"

    values = [Fraction(0)] * n_total
    for i, b in enumerate(basis):
        if b < n_total:
            values[b] = tableau[i][1]

    x = []
    for j in range(num_vars):
        v = values[var_pos[j]]
        if j in var_neg:
            v -= values[var_neg[j]]
        x.append(v)

    objective = sum(
        (obj.get(j, Fraction(0)) * x[j] for j in range(num_vars)), Fraction(0)
    )
    return LPResult(status="optimal", x=x, objective=objective)


def _pivot_row(row, rhs, col):
    pivot = row[col]
    new_row = {c: v / pivot for c, v in row.items() if v != 0}
    new_row.pop(col, None)
    new_row[col] = Fraction(1)
    return new_row, rhs / pivot


def _eliminate_column(tableau, pivot_i, col):
    """Zeroes out `col` in every row but `pivot_i`, mutating `tableau` in place."""
    pivot_row, pivot_rhs = tableau[pivot_i]
    for i, (row, rhs) in enumerate(tableau):
        if i == pivot_i:
            continue
        factor = row.get(col, Fraction(0))
        if factor == 0:
            continue
        new_row = dict(row)
        for c, v in pivot_row.items():
            new_value = new_row.get(c, Fraction(0)) - factor * v
            if new_value == 0:
                new_row.pop(c, None)
            else:
                new_row[c] = new_value
        tableau[i] = (new_row, rhs - factor * pivot_rhs)


def _reduced_costs(tableau, basis, obj):
    # c_bar_j = c_j - sum_i (c_{basis[i]} * row_i[j]) for every non-basic j.
    reduced = dict(obj)
    for i, b in enumerate(basis):
        cb = obj.get(b, Fraction(0))
        if cb == 0:
            continue
        row, _ = tableau[i]
        for c, v in row.items():
            reduced[c] = reduced.get(c, Fraction(0)) - cb * v
    return reduced


def _objective_value(tableau, basis, obj):
    return sum(
        (obj.get(b, Fraction(0)) * rhs for b, (_, rhs) in zip(basis, tableau)),
        Fraction(0),
    )


def _run_simplex(tableau, basis, obj, n_cols, max_iters=100000, progress=None,
                  stall_threshold=None):
    """Minimizes `obj` over `tableau` in place.

    Sequence-form game LPs are highly degenerate (many ties in the ratio
    test), and plain Bland's rule -- while provably non-cycling -- picks
    its entering column by lowest index regardless of how much progress it
    makes, which is asymptotically fine but can be catastrophically slow in
    practice on a degenerate LP with thousands of columns. So the entering
    column normally uses Dantzig's rule (most negative reduced cost, which
    converges fast in practice) and only falls back to strict Bland's rule
    -- guaranteed to terminate -- once a run of degenerate (zero-progress)
    pivots signals the problem has stalled.

    The reduced-cost row is maintained incrementally (updated by the same
    elimination applied to each pivot) rather than recomputed from scratch
    every iteration -- for LPs with hundreds of rows, recomputing it fresh
    each time roughly doubles the total work for no benefit.
    """
    if stall_threshold is None:
        stall_threshold = max(50, 2 * len(tableau))
    reduced = _reduced_costs(tableau, basis, obj)
    basic_set = set(basis)
    iteration = 0
    stall_count = 0
    use_bland = False
    while True:
        entering = None
        if use_bland:
            # Bland's rule: enter the lowest-indexed column with negative
            # reduced cost. Guarantees termination.
            for c in range(n_cols):
                if c in basic_set:
                    continue
                if reduced.get(c, Fraction(0)) < 0:
                    entering = c
                    break
        else:
            # Dantzig's rule: enter the column with the most negative
            # reduced cost.
            best_reduced = Fraction(0)
            for c in range(n_cols):
                if c in basic_set:
                    continue
                rc = reduced.get(c, Fraction(0))
                if rc < best_reduced:
                    best_reduced = rc
                    entering = c
        if entering is None:
            return "optimal"

        # Ratio test, ties broken by lowest-indexed basic variable
        # (Bland's rule) to prevent cycling.
        leaving_i = None
        best_ratio = None
        for i, (row, rhs) in enumerate(tableau):
            a = row.get(entering, Fraction(0))
            if a <= 0:
                continue
            ratio = rhs / a
            if (
                best_ratio is None
                or ratio < best_ratio
                or (ratio == best_ratio and basis[i] < basis[leaving_i])
            ):
                best_ratio = ratio
                leaving_i = i
        if leaving_i is None:
            raise RuntimeError("LP is unbounded")

        if best_ratio == 0:
            stall_count += 1
            if not use_bland and stall_count >= stall_threshold:
                use_bland = True
        else:
            stall_count = 0

        row, rhs = tableau[leaving_i]
        tableau[leaving_i] = _pivot_row(row, rhs, entering)
        pivot_row, pivot_rhs = tableau[leaving_i]
        basic_set.discard(basis[leaving_i])
        basis[leaving_i] = entering
        basic_set.add(entering)
        _eliminate_column(tableau, leaving_i, entering)

        # Update the reduced-cost row with the same elimination step.
        factor = reduced.get(entering, Fraction(0))
        if factor != 0:
            for c, v in pivot_row.items():
                new_value = reduced.get(c, Fraction(0)) - factor * v
                if new_value == 0:
                    reduced.pop(c, None)
                else:
                    reduced[c] = new_value

        iteration += 1
        if iteration >= max_iters:
            raise RuntimeError("simplex did not converge")
        if progress is not None and iteration % progress.get("every", 50) == 0:
            progress["callback"](iteration, tableau, use_bland, stall_count)
