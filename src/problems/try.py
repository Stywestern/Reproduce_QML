"""
Data + math backend for Stochastic Unit Commitment (SUC), decoupled from any
solving algorithm. This module answers ONE question: "what is the problem?"
-- not "how do we solve it?" Every downstream tool (classical MILP, Benders
master/subproblem split, binary-encoding preprocessing for QUBO methods,
ADMM block partitioning) should be able to consume the MixedLBOProblem this
module produces without knowing anything about SUC specifically.

Layers:
    SUCProblemData     -- scalable technical + scenario data (N/T/S all free)
    LBOBuilder / MixedLBOProblem -- generic (c, A, row_lower, row_upper)
                          representation + incremental builder. Sparse by
                          construction, since dense matrices don't survive
                          contact with realistic problem sizes (bus-118 with
                          100 scenarios is ~130K variables / ~260K rows).
    SUCModelBuilder    -- SUC-specific: variables, objective, and ONLY the
                          "laws of physics" (power balance, capacity limits).
                          Deliberately leaves the builder open afterwards so
                          structural constraints (min up/down time) or
                          algorithmic artifacts (Benders cuts) can be layered
                          on top without touching this file.

Extending this: add_min_up_down_time_constraints() below is one example of
that extension pattern -- it takes an already-built SUCModelBuilder and adds
rows to its (still-open) internal LBOBuilder, rather than being baked into
setup_core_constraints(). Benders' cut-accumulation and ADMM's block
partitioning should follow the same pattern in their own modules.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import scipy.sparse as sp


# ==========================================================================
# 1. Problem data -- fully parameterized so N/T/S can all be changed, with
#    validation so a malformed override array fails immediately and
#    legibly instead of deep inside LBOBuilder.build().
# ==========================================================================

@dataclass
class SUCProblemData:
    num_gens: int               # N
    num_hours: int               # T
    num_scenarios: int           # S
    p_min: np.ndarray            # (N,) MW
    p_max: np.ndarray            # (N,) MW
    c_var: np.ndarray            # (N,) $/MWh
    c_fixed: np.ndarray          # (N,) $/h
    min_up: np.ndarray           # (N,) integer hours, T^U_g
    min_down: np.ndarray         # (N,) integer hours, T^D_g
    scenario_probs: np.ndarray   # (S,), sums to 1
    demand: np.ndarray           # (S, T)
    wind: np.ndarray             # (S, T)
    shed_cost: float = 1000.0    # $/MWh penalty for unmet demand

    def __post_init__(self):
        N, T, S = self.num_gens, self.num_hours, self.num_scenarios
        assert self.p_min.shape == (N,) and self.p_max.shape == (N,)
        assert np.all(self.p_min <= self.p_max), "p_min must be <= p_max for every generator"
        assert self.c_var.shape == (N,) and self.c_fixed.shape == (N,)
        assert self.min_up.shape == (N,) and self.min_down.shape == (N,)
        assert np.all(self.min_up >= 1) and np.all(self.min_down >= 1)
        assert self.scenario_probs.shape == (S,)
        assert abs(self.scenario_probs.sum() - 1.0) < 1e-9, "scenario_probs must sum to 1"
        assert self.demand.shape == (S, T) and self.wind.shape == (S, T)

    @classmethod
    def generate(cls, num_gens: int, num_hours: int, num_scenarios: int, seed: int = 2,
                 p_min: np.ndarray | None = None,
                 p_max: np.ndarray | None = None,
                 c_var: np.ndarray | None = None,
                 c_fixed: np.ndarray | None = None,
                 min_up: np.ndarray | None = None,
                 min_down: np.ndarray | None = None,
                 scenario_probs: np.ndarray | None = None,
                 demand: np.ndarray | None = None,
                 wind: np.ndarray | None = None,
                 shed_cost: float | None = None) -> "SUCProblemData":
        """
        Builds problem data by prioritizing user-supplied arrays. Any array
        left as None is synthesized from the scaling-benchmark rules below.
        """
        rng = np.random.default_rng(seed)
        tier = np.linspace(0, 1, num_gens)
        t = np.arange(num_hours)

        actual_p_max = p_max if p_max is not None else np.clip(
            150 - 100 * tier + rng.normal(0, 5, num_gens), 10, None)
        actual_p_min = p_min if p_min is not None else 0.15 * actual_p_max

        actual_c_var = c_var if c_var is not None else 15 + 35 * tier
        actual_c_fixed = c_fixed if c_fixed is not None else 500 - 400 * tier
        actual_shed_cost = shed_cost if shed_cost is not None else 1000.0

        actual_min_up = min_up if min_up is not None else np.clip(
            (6 - 4 * tier).round().astype(int), 1, None)
        actual_min_down = min_down if min_down is not None else actual_min_up.copy()

        actual_probs = scenario_probs if scenario_probs is not None else np.full(
            num_scenarios, 1.0 / num_scenarios)

        if demand is not None:
            actual_demand = demand
        else:
            base_demand = (0.4 * actual_p_max.sum()) + 0.15 * actual_p_max.sum() * np.sin(
                (t - num_hours / 4) * np.pi / max(num_hours / 2, 1)
            )
            actual_demand = np.stack([
                base_demand + (rng.beta(2, 5, num_hours) * 40 - 15)
                for _ in range(num_scenarios)
            ])

        if wind is not None:
            actual_wind = wind
        else:
            actual_wind = np.stack([
                np.clip((rng.weibull(2, num_hours) * 10 / 12) ** 3 * 60, 0, 60)
                for _ in range(num_scenarios)
            ])

        return cls(
            num_gens=num_gens, num_hours=num_hours, num_scenarios=num_scenarios,
            p_min=actual_p_min, p_max=actual_p_max, c_var=actual_c_var,
            c_fixed=actual_c_fixed, min_up=actual_min_up, min_down=actual_min_down,
            scenario_probs=actual_probs, demand=actual_demand, wind=actual_wind,
            shed_cost=actual_shed_cost,
        )


# ==========================================================================
# 2. Mathematical backend -- generic (c, A, row_lower, row_upper) form.
#    Not SUC-specific; any decomposition/solver tool consumes this.
# ==========================================================================

@dataclass
class MixedLBOProblem:
    """
    min c^T x  s.t.  row_lower <= A x <= row_upper,  x_i in [lower_i, upper_i],
    x_i binary if var_types[i] == 'B', continuous if 'C'.
    """
    
    var_names: list[str]
    var_types: list[str]         # 'B' or 'C', one per variable
    lower: np.ndarray
    upper: np.ndarray
    c: np.ndarray
    A: sp.csr_matrix
    row_lower: np.ndarray
    row_upper: np.ndarray
    constraint_labels: list[str]

    def var_index(self, name: str) -> int:
        return self.var_names.index(name)


class LBOBuilder:
    """
    Incremental builder: register variables, then add objective terms and
    constraint rows by NAME (not index), then .build() materializes the
    sparse matrices. Rows are accumulated as (name -> coeff) dicts and only
    assembled into a COO/CSR matrix at build() time -- O(nnz), not O(m*n).
    """

    def __init__(self):
        self._names: list[str] = []
        self._types: list[str] = []
        self._lower: list[float] = []
        self._upper: list[float] = []
        self._index: dict[str, int] = {}
        self._c: dict[str, float] = {}
        self._rows: list[dict[str, float]] = []
        self._senses: list[str] = []
        self._rhs: list[float] = []
        self._labels: list[str] = []

    def add_var(self, name: str, vtype: str, lower: float = 0.0, upper: float = 1.0) -> None:
        assert name not in self._index, f"duplicate variable {name}"
        assert vtype in ("B", "C"), f"unknown variable type {vtype!r}"
        self._index[name] = len(self._names)
        self._names.append(name)
        self._types.append(vtype)
        self._lower.append(lower)
        self._upper.append(upper)

    def add_objective_term(self, name: str, coeff: float) -> None:
        self._c[name] = self._c.get(name, 0.0) + coeff

    def add_constraint(self, coeffs: dict[str, float], rhs: float,
                        sense: str = "<=", label: str = "") -> None:
        """Adds one row: sum(coeffs[v]*v) {sense} rhs. sense in {'<=','>=','='}."""
        assert sense in {"<=", ">=", "="}, f"unknown sense {sense!r}"
        self._rows.append(coeffs)
        self._senses.append(sense)
        self._rhs.append(rhs)
        self._labels.append(label)

    def build(self) -> MixedLBOProblem:
        n = len(self._names)
        c = np.zeros(n)
        for name, coeff in self._c.items():
            c[self._index[name]] = coeff

        m = len(self._rows)
        row_idx, col_idx, data = [], [], []
        row_lower = np.empty(m)
        row_upper = np.empty(m)
        for i, (row, sense, rhs) in enumerate(zip(self._rows, self._senses, self._rhs)):
            for name, coeff in row.items():
                if coeff == 0.0:
                    continue
                row_idx.append(i)
                col_idx.append(self._index[name])
                data.append(coeff)
            if sense == "<=":
                row_lower[i], row_upper[i] = -np.inf, rhs
            elif sense == ">=":
                row_lower[i], row_upper[i] = rhs, np.inf
            else:  # "="
                row_lower[i], row_upper[i] = rhs, rhs

        A = sp.coo_matrix((data, (row_idx, col_idx)), shape=(m, n)).tocsr()

        return MixedLBOProblem(
            var_names=self._names, var_types=self._types,
            lower=np.array(self._lower), upper=np.array(self._upper),
            c=c, A=A, row_lower=row_lower, row_upper=row_upper,
            constraint_labels=self._labels,
        )


# ==========================================================================
# 3. SUC model formulation -- variables, objective, and ONLY the physics.
#    Structural constraints (min up/down time) and algorithmic constraints
#    (Benders cuts, ADMM consensus) are added on top by OTHER modules,
#    operating on the still-open self.builder -- see
#    add_min_up_down_time_constraints() below for the pattern.
# ==========================================================================

class SUCModelBuilder:
    def __init__(self, problem_data: SUCProblemData):
        self.pd = problem_data
        self.builder = LBOBuilder()

    def _u_name(self, g: int, t: int) -> str:
        return f"u_gen{g}_hr{t}"

    def _p_name(self, g: int, t: int, s: int) -> str:
        return f"P_gen{g}_hr{t}_scen{s}"

    def _shed_name(self, t: int, s: int) -> str:
        return f"shed_hr{t}_scen{s}"

    def setup_variables(self) -> None:
        N, T, S = self.pd.num_gens, self.pd.num_hours, self.pd.num_scenarios
        # Upper bound for shed derived from the data itself, not a magic
        # constant -- shedding can never usefully exceed the largest single
        # net demand actually observed across all scenarios/hours.
        shed_upper = float(self.pd.demand.max()) if self.pd.demand.size else 1.0

        for g in range(N):
            for t in range(T):
                self.builder.add_var(self._u_name(g, t), "B", lower=0.0, upper=1.0)

        for s in range(S):
            for t in range(T):
                self.builder.add_var(self._shed_name(t, s), "C", lower=0.0, upper=shed_upper)
                for g in range(N):
                    self.builder.add_var(self._p_name(g, t, s), "C",
                                          lower=0.0, upper=self.pd.p_max[g])

    def setup_objective(self) -> None:
        N, T, S = self.pd.num_gens, self.pd.num_hours, self.pd.num_scenarios

        for g in range(N):
            for t in range(T):
                self.builder.add_objective_term(self._u_name(g, t), self.pd.c_fixed[g])

        for s in range(S):
            prob = self.pd.scenario_probs[s]
            expected_shed_cost = prob * self.pd.shed_cost  # hoisted: t-independent
            for t in range(T):
                self.builder.add_objective_term(self._shed_name(t, s), expected_shed_cost)
                for g in range(N):
                    self.builder.add_objective_term(
                        self._p_name(g, t, s), prob * self.pd.c_var[g])

    def setup_core_constraints(self) -> None:
        N, T, S = self.pd.num_gens, self.pd.num_hours, self.pd.num_scenarios

        for s in range(S):
            for t in range(T):
                # --- A. Power balance: sum(P) + shed == net_demand ---
                net_demand = self.pd.demand[s, t] - self.pd.wind[s, t]
                coeffs = {self._shed_name(t, s): 1.0}
                for g in range(N):
                    coeffs[self._p_name(g, t, s)] = 1.0
                self.builder.add_constraint(coeffs, rhs=net_demand, sense="=",
                                             label=f"balance_s{s}_t{t}")

                # --- B. Generator coupling limits ---
                for g in range(N):
                    u_var = self._u_name(g, t)
                    p_var = self._p_name(g, t, s)

                    # P <= p_max * u  -->  P - p_max*u <= 0
                    self.builder.add_constraint(
                        {p_var: 1.0, u_var: -self.pd.p_max[g]},
                        rhs=0.0, sense="<=", label=f"upper_s{s}_t{t}_g{g}")

                    # P >= p_min * u  -->  -P + p_min*u <= 0
                    self.builder.add_constraint(
                        {p_var: -1.0, u_var: self.pd.p_min[g]},
                        rhs=0.0, sense="<=", label=f"lower_s{s}_t{t}_g{g}")

    def get_base_model(self) -> MixedLBOProblem:
        """Runs the setup and returns the compiled mathematical matrices.
        Call this AFTER any extension functions (e.g.
        add_min_up_down_time_constraints) if you want their rows included --
        or call it first and extend a separately-tracked builder, since
        self.builder stays open either way."""
        self.setup_variables()
        self.setup_objective()
        self.setup_core_constraints()
        return self.builder.build()


# ==========================================================================
# 4. Example extension: minimum up/down time. Demonstrates the intended
#    pattern for layering structural/algorithmic constraints on top of the
#    base model WITHOUT modifying SUCModelBuilder itself. Benders' cut
#    accumulation and ADMM's block partitioning should follow this same
#    shape in their own modules.
# ==========================================================================

def add_min_up_down_time_constraints(model: SUCModelBuilder) -> None:
    """
    Adds eq. (2d)-(2e)-style constraints directly to model.builder (still
    open -- must be called before model.get_base_model()). u[g,-1] = 0 is
    assumed (all units start OFF); the enforcement window is correctly
    CAPPED (not dropped) at T-1 for start times near the end of the horizon.
    """
    pd = model.pd
    N, T = pd.num_gens, pd.num_hours
    b = model.builder

    for g in range(N):
        Lu, Ld = int(pd.min_up[g]), int(pd.min_down[g])
        for t in range(T):
            u_t = model._u_name(g, t)
            u_prev = model._u_name(g, t - 1) if t > 0 else None

            up_end = min(t + Lu - 1, T - 1)
            for tau in range(t, up_end + 1):
                # startup - u[tau] <= 0, startup = u[t] - u[prev]
                coeffs = {u_t: 1.0, model._u_name(g, tau): -1.0}
                if u_prev is not None:
                    coeffs[u_prev] = coeffs.get(u_prev, 0.0) - 1.0
                b.add_constraint(coeffs, rhs=0.0, sense="<=",
                                  label=f"minup_g{g}_t{t}_tau{tau}")

            down_end = min(t + Ld - 1, T - 1)
            for tau in range(t, down_end + 1):
                # shutdown - (1 - u[tau]) <= 0, shutdown = u[prev] - u[t]
                # => u[prev] - u[t] + u[tau] <= 1
                coeffs = {u_t: -1.0, model._u_name(g, tau): 1.0}
                if u_prev is not None:
                    coeffs[u_prev] = coeffs.get(u_prev, 0.0) + 1.0
                b.add_constraint(coeffs, rhs=1.0, sense="<=",
                                  label=f"mindown_g{g}_t{t}_tau{tau}")