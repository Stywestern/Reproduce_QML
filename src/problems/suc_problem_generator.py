############################################################################################
#                      PROBLEM: 4-GENERATOR STOCHASTIC UC MODEL
############################################################################################

# Third Party Libraries
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable
import numpy as np
from scipy.optimize import linprog, milp, LinearConstraint, Bounds

"""
Modular pipeline for reproducing "Qubit-Efficient Quantum Annealing for
Stochastic Unit Commitment" (Hong, Xu, Teng, IEEE TPWRS 2026).

Core idea: every algorithm in the paper (classical MILP, Basic QA, QPHR-ALM,
QPHR-ADMM) ultimately wants the SAME thing -- a compact standard-form LP/MIP,
min c^T x s.t. A x <= b, with typed/bounded variables -- and differs only in
what PREPROCESSING it applies before solving:

    build master (SUCProblemData -> MixedLBOProblem)
            |
            +-- classical MILP: use as-is (mixed binary/continuous)
            |
            +-- QUBO-based (Basic QA / QPHR-ALM / D-ADMM):
                    binary_encode_continuous(Upsilon)  -> all-binary problem
                    [Basic QA only: also add_binary_slacks() per constraint]
                    -> algorithm-specific QUBO construction (separate module,
                       not yet implemented)

Layers:
    SUCProblemData        -- scalable technical + scenario data (N, T, K free)
    DispatchSubproblem     -- per-scenario continuous LP given fixed u;
                              returns cost + duals (Benders cut ingredients)
    LBOBuilder / MixedLBOProblem -- generic (c, A, b) representation + builder
    binary_encode_continuous     -- generic preprocessing transform
    BendersMasterBuilder    -- SUC-specific: emits the master as a MixedLBOProblem
    MasterSolver (ABC)       -- common interface every backend implements
        ClassicalMILPSolver  -- fully implemented, scipy.optimize.milp
        (QUBO-based backends: next step, see chat)
"""

# ==========================================================================
# 1. Problem data -- fully parameterized so N/T/K can all be changed. One can even add more constraints with some work.
# ==========================================================================

@dataclass
class SUCProblemData:
    num_gens: int              # N
    num_hours: int             # T
    num_scenarios: int         # K
    p_min: np.ndarray          # (N,), p = power of generator in MW
    p_max: np.ndarray          # (N,)
    c_var: np.ndarray          # (N,)  $/MWh, c = cost of generators variable output
    c_fixed: np.ndarray        # (N,)  $/h, cost of just running it on
    min_up: np.ndarray         # (N,) integer hours, T^U_g, a generator must stay on for this much hours when opened
    min_down: np.ndarray       # (N,) integer hours, T^D_g
    scenario_probs: np.ndarray  # (K,), sums to 1
    demand: np.ndarray         # (K, T), generators must met this
    wind: np.ndarray           # (K, T), eases the generator load
    shed_cost: float = 1000.0  # If in any scenario we have to cut electricity from people

    @classmethod
    def generate(cls, num_gens: int, num_hours: int, num_scenarios: int,seed: int = 2,
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
        Builds the problem data by prioritizing user-supplied arrays. 
        If a specific array is not provided (is None), it generates a synthetic 
        version of it based on the scaling benchmark rules.
        """

        rng = np.random.default_rng(seed)
        tier = np.linspace(0, 1, num_gens)
        t = np.arange(num_hours)

        # 1. Generator Capacities
        if p_max is not None:
            actual_p_max = p_max
        else:
            actual_p_max = np.clip(150 - 100 * tier + rng.normal(0, 5, num_gens), 10, None)
            
        if p_min is not None:
            actual_p_min = p_min
        else:
            actual_p_min = 0.15 * actual_p_max

        # 2. Generator Costs
        actual_c_var = c_var if c_var is not None else 15 + 35 * tier
        actual_c_fixed = c_fixed if c_fixed is not None else 500 - 400 * tier
        actual_shed_cost = shed_cost if shed_cost is not None else 1000.0

        # 3. Operational Timings
        if min_up is not None:
            actual_min_up = min_up
        else:
            actual_min_up = np.clip((6 - 4 * tier).round().astype(int), 1, None)
            
        if min_down is not None:
            actual_min_down = min_down
        else:
            actual_min_down = actual_min_up.copy()

        # 4. Scenarios & Weather Data
        actual_probs = scenario_probs if scenario_probs is not None else np.full(num_scenarios, 1.0 / num_scenarios)

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

        # 5. Return the finalized dataclass object
        return cls(
            num_gens=num_gens, 
            num_hours=num_hours, 
            num_scenarios=num_scenarios, 
            p_min=actual_p_min, 
            p_max=actual_p_max, 
            c_var=actual_c_var,
            c_fixed=actual_c_fixed, 
            min_up=actual_min_up, 
            min_down=actual_min_down, 
            scenario_probs=actual_probs, 
            demand=actual_demand, 
            wind=actual_wind,
            shed_cost=actual_shed_cost
        )


# ==========================================================================
# 2. Dispatch subproblem: per-scenario continuous LP given FIXED u.
#    Fully implemented; independent of any quantum machinery.
# ==========================================================================

@dataclass
class DispatchResult:
    cost: float              # total cost in dollars
    P: np.ndarray            # (N, T), generated power of every gen for every time instance
    shed: np.ndarray         # (T,)
    theta: np.ndarray        # (N, T) -- d(cost)/d(u[g,t]), the Benders dual


class DispatchSubproblem:
    """
    For ONE scenario h and a FIXED commitment u (shape N x T), solves:
        min  sum_{g,t} c_var[g]*P[g,t] + shed_cost * sum_t shed[t]
        s.t. P[g,t] - p_max[g]*u[g,t] <= 0                (upper coupling)
             p_min[g]*u[g,t] - P[g,t] <= 0                (lower coupling)
             sum_g P[g,t] + shed[t] = net_demand[t]        (power balance)
             P, shed >= 0

    theta[g,t] = p_max[g]*mu_upper[g,t] - p_min[g]*mu_lower[g,t], by the
    envelope theorem applied to the two u-dependent bound constraints --
    this is exactly theta_{g,t}(xi_h) in eq. (2c)/(3e) of the paper.

    NOTE: validate this sign convention against the hand-solved single-hour
    examples (mu_A=5 etc.) before trusting it on a new instance -- scipy's
    HiGHS marginal-sign convention should be treated as an assumption here,
    not a certainty, until cross-checked.
    """

    def __init__(self, problem: SUCProblemData):
        self.pd = problem

    def solve(self, u: np.ndarray, scenario_idx: int) -> DispatchResult:
        N, T = self.pd.num_gens, self.pd.num_hours
        s = scenario_idx
        net_demand = self.pd.demand[s] - self.pd.wind[s]

        # Variable order: P[g,t] flattened (g-major), then shed[t]
        n_P = N * T
        n_x = n_P + T

        def p_idx(g, t):
            return g * T + t

        def shed_idx(t):
            return n_P + t

        c = np.zeros(n_x)
        for g in range(N):
            for t in range(T):
                c[p_idx(g, t)] = self.pd.c_var[g]
        c[n_P:] = self.pd.shed_cost

        # Inequality constraints A_ub x <= b_ub : the two u-coupling bounds
        rows_ub, b_ub = [], []
        for g in range(N):
            for t in range(T):
                # P[g,t] - p_max[g]*u[g,t] <= 0
                row = np.zeros(n_x)
                row[p_idx(g, t)] = 1.0
                rows_ub.append(row)
                b_ub.append(self.pd.p_max[g] * u[g, t])
                # p_min[g]*u[g,t] - P[g,t] <= 0  ->  -P[g,t] <= -p_min*u
                row2 = np.zeros(n_x)
                row2[p_idx(g, t)] = -1.0
                rows_ub.append(row2)
                b_ub.append(-self.pd.p_min[g] * u[g, t])
        A_ub = np.array(rows_ub)
        b_ub = np.array(b_ub)

        # Equality: sum_g P[g,t] + shed[t] = net_demand[t]
        A_eq = np.zeros((T, n_x))
        for t in range(T):
            for g in range(N):
                A_eq[t, p_idx(g, t)] = 1.0
            A_eq[t, shed_idx(t)] = 1.0
        b_eq = net_demand

        bounds = [(0, None)] * n_x

        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                       bounds=bounds, method="highs")
        if not res.success:
            raise RuntimeError(f"Dispatch subproblem infeasible for scenario {s}: {res.message}")

        P = res.x[:n_P].reshape(N, T)
        shed = res.x[n_P:]

        # Duals on the two coupling rows per (g,t); HiGHS orders A_ub rows as we built them: pairs (upper, lower) per (g,t) in insertion order.
        marg = res.ineqlin.marginals  # shape (2*N*T,), sign per scipy convention
        theta = np.zeros((N, T))
        row = 0
        for g in range(N):
            for t in range(T):
                mu_upper = -marg[row]      # flip sign: marginal on "<=" is <=0 in HiGHS
                mu_lower = -marg[row + 1]
                theta[g, t] = self.pd.p_max[g] * mu_upper - self.pd.p_min[g] * mu_lower
                row += 2

        return DispatchResult(cost=res.fun, P=P, shed=shed, theta=theta)


# ==========================================================================
# 3. Generic (c, A, b) layer -- solver-agnostic, NOT SUC-specific.
# ==========================================================================

@dataclass
class MixedLBOProblem:
    """min c^T x  s.t.  A x <= b,  x_i in [lower_i, upper_i],
    x_i binary if var_types[i] == 'B', continuous if 'C'.
    This is the ONE object every solver backend consumes."""
    var_names: list[str]
    var_types: list[str]      # 'B' or 'C', one per variable
    lower: np.ndarray
    upper: np.ndarray
    c: np.ndarray
    A: np.ndarray
    b: np.ndarray
    constraint_labels: list[str]

    def var_index(self, name: str) -> int:
        return self.var_names.index(name)


class LBOBuilder:
    """Incremental builder: register variables, then add objective terms and
    constraint rows by NAME (not index), then .build() materializes the
    matrices. Keeps BendersMasterBuilder's code readable."""

    def __init__(self):
        self._names: list[str] = []
        self._types: list[str] = []
        self._lower: list[float] = []
        self._upper: list[float] = []
        self._index: dict[str, int] = {}
        self._c: dict[str, float] = {}
        self._rows: list[dict[str, float]] = []
        self._rhs: list[float] = []
        self._labels: list[str] = []

    def add_var(self, name: str, vtype: str, lower: float = 0.0, upper: float = 1.0) -> None:
        assert name not in self._index, f"duplicate variable {name}"
        self._index[name] = len(self._names)
        self._names.append(name)
        self._types.append(vtype)
        self._lower.append(lower)
        self._upper.append(upper)

    def add_objective_term(self, name: str, coeff: float) -> None:
        self._c[name] = self._c.get(name, 0.0) + coeff

    def add_constraint(self, coeffs: dict[str, float], rhs: float, label: str = "") -> None:
        """Adds one row: sum(coeffs[v]*v) <= rhs."""
        self._rows.append(coeffs)
        self._rhs.append(rhs)
        self._labels.append(label)

    def build(self) -> MixedLBOProblem:
        n = len(self._names)
        c = np.zeros(n)
        for name, coeff in self._c.items():
            c[self._index[name]] = coeff

        m = len(self._rows)
        A = np.zeros((m, n))
        for i, row in enumerate(self._rows):
            for name, coeff in row.items():
                A[i, self._index[name]] = coeff
        b = np.array(self._rhs)

        return MixedLBOProblem(
            var_names=self._names, var_types=self._types,
            lower=np.array(self._lower), upper=np.array(self._upper),
            c=c, A=A, b=b, constraint_labels=self._labels,
        )


def binary_encode_continuous(problem: MixedLBOProblem, var_name: str,
                              num_bits: int, scale: float) -> MixedLBOProblem:
    """
    Generic preprocessing transform (paper's eq. under III-A/1): replaces
    ONE continuous variable y (0 <= y <= scale*(2^num_bits - 1)) with
    num_bits new binary variables y_j, via y = scale * sum_j 2^j * y_j.
    Substitutes this into the objective and every constraint row, then
    drops the original continuous column. Works on ANY MixedLBOProblem,
    not just the SUC master's Upsilon -- reusable preprocessing step.
    """
    idx = problem.var_index(var_name)
    assert problem.var_types[idx] == "C", f"{var_name} is not continuous"

    new_names = [n for i, n in enumerate(problem.var_names) if i != idx]
    bit_names = [f"{var_name}_bit{j}" for j in range(num_bits)]
    new_names = new_names + bit_names

    new_types = [t for i, t in enumerate(problem.var_types) if i != idx] + ["B"] * num_bits
    new_lower = np.concatenate([np.delete(problem.lower, idx), np.zeros(num_bits)])
    new_upper = np.concatenate([np.delete(problem.upper, idx), np.ones(num_bits)])

    keep_mask = np.ones(len(problem.var_names), dtype=bool)
    keep_mask[idx] = False

    weights = scale * (2.0 ** np.arange(num_bits))

    # Objective: c_y * y  ->  sum_j c_y * weight_j * y_bit_j
    c_y = problem.c[idx]
    new_c = np.concatenate([problem.c[keep_mask], c_y * weights])

    # Constraints: A[:, idx] * y  ->  A[:, idx] contributes weight_j * A[:,idx] to each bit col
    A_kept = problem.A[:, keep_mask]
    A_y = problem.A[:, idx:idx + 1]                 # (m, 1)
    A_bits = A_y @ weights.reshape(1, -1)            # (m, num_bits)
    new_A = np.hstack([A_kept, A_bits])

    return MixedLBOProblem(
        var_names=new_names, var_types=new_types,
        lower=new_lower, upper=new_upper,
        c=new_c, A=new_A, b=problem.b.copy(),
        constraint_labels=list(problem.constraint_labels),
    )


# ==========================================================================
# 4. SUC-specific: master problem builder.
# ==========================================================================

class BendersMasterBuilder:
    """
    Owns the master's structural constraints (min up/down time, fixed once)
    and its growing cut set (one row added per Benders iteration).
    emit() returns the CURRENT master as a MixedLBOProblem -- u binary,
    Upsilon continuous, ready either for a classical MILP solver as-is, or
    for binary_encode_continuous(..., "Upsilon", ...) before a QUBO backend.
    """

    def __init__(self, problem: SUCProblemData, upsilon_upper_bound: float):
        self.pd = problem
        self.upsilon_upper = upsilon_upper_bound
        self._cuts: list[dict] = []  # each: {"coeffs": {...}, "rhs": float}

    def _u_name(self, g: int, t: int) -> str:
        return f"u_{g}_{t}"

    def add_optimality_cut(self, u_bar: np.ndarray,
                            per_scenario_theta: np.ndarray,
                            per_scenario_cost: np.ndarray) -> None:
        """
        per_scenario_theta: (K, N, T) -- theta[h, g, t] from DispatchSubproblem
        per_scenario_cost:  (K,)      -- Q_h(u_bar) from DispatchSubproblem

        Implements eq. (2c):
            Upsilon >= sum_h p_h [Q_h(u_bar) + sum_{g,t} theta_{g,t}(h) (u[g,t] - u_bar[g,t])]
        Rearranged to standard "<=" form:
            -Upsilon + sum_{g,t} T[g,t] * u[g,t] <= -RHS_const
        where T[g,t] = sum_h p_h * theta[h,g,t]  (aggregated dual)
        """
        p = self.pd.scenario_probs
        N, T = self.pd.num_gens, self.pd.num_hours

        T_agg = np.tensordot(p, per_scenario_theta, axes=(0, 0))  # (N, T)
        rhs_const = float(np.dot(p, per_scenario_cost)
                           - sum(p[h] * np.sum(per_scenario_theta[h] * u_bar)
                                 for h in range(len(p))))

        coeffs = {"Upsilon": -1.0}
        for g in range(N):
            for t in range(T):
                if T_agg[g, t] != 0.0:
                    coeffs[self._u_name(g, t)] = T_agg[g, t]

        self._cuts.append({"coeffs": coeffs, "rhs": -rhs_const})

    def emit(self) -> MixedLBOProblem:
        pd = self.pd
        N, T = pd.num_gens, pd.num_hours
        b = LBOBuilder()

        for g in range(N):
            for t in range(T):
                b.add_var(self._u_name(g, t), "B", 0, 1)
        b.add_var("Upsilon", "C", 0, self.upsilon_upper)

        for g in range(N):
            for t in range(T):
                b.add_objective_term(self._u_name(g, t), pd.c_fixed[g])
        b.add_objective_term("Upsilon", 1.0)

        # Minimum up/down time -- u[g,-1] = 0 assumed; tail window capped
        # (not dropped) at T-1, matching the earlier corrected derivation.
        for g in range(N):
            Lu, Ld = int(pd.min_up[g]), int(pd.min_down[g])
            for t in range(T):
                u_t = self._u_name(g, t)
                u_prev = self._u_name(g, t - 1) if t > 0 else None

                up_end = min(t + Lu - 1, T - 1)
                for tau in range(t, up_end + 1):
                    # startup - u[tau] <= 0, startup = u[t] - u[prev]
                    coeffs = {u_t: 1.0, self._u_name(g, tau): -1.0}
                    if u_prev is not None:
                        coeffs[u_prev] = coeffs.get(u_prev, 0.0) - 1.0
                    b.add_constraint(coeffs, 0.0, label=f"minup_g{g}_t{t}_tau{tau}")

                down_end = min(t + Ld - 1, T - 1)
                for tau in range(t, down_end + 1):
                    # shutdown - (1 - u[tau]) <= 0, shutdown = u[prev] - u[t]
                    # => u[prev] - u[t] + u[tau] <= 1
                    coeffs = {u_t: -1.0, self._u_name(g, tau): 1.0}
                    if u_prev is not None:
                        coeffs[u_prev] = coeffs.get(u_prev, 0.0) + 1.0
                    b.add_constraint(coeffs, 1.0, label=f"mindown_g{g}_t{t}_tau{tau}")

        for i, cut in enumerate(self._cuts):
            b.add_constraint(cut["coeffs"], cut["rhs"], label=f"cut_{i}")

        return b.build()


############################################################################################
#                                   Execution Block
############################################################################################
if __name__ == "__main__":
    import os

    # Ensure log.txt appears in the exact folder where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else "."
    log_path = os.path.join(script_dir, "4gen_suc_problem.txt")

    # Instantiate Problem with 4-hour minimum up/down time parameter
    problem = FourGeneratorUCModel(num_hours=24, num_scenarios=3, min_up_time=4, seed=42)
    cqm = problem.get_cqm()

    binary_vars = [v for v in cqm.variables if cqm.vartype(v) == dimod.BINARY]
    real_vars = [v for v in cqm.variables if cqm.vartype(v) == dimod.REAL]
    num_constraints = len(cqm.constraints)

    # 1. Write everything into log.txt
    with open(log_path, "w", encoding="utf-8") as log_file:
        
        def log_write(text=""):
            log_file.write(text + "\n")

        log_write("==========================================================================")
        log_write("         STOCHASTIC 4-GENERATOR UNIT COMMITMENT MODEL INSPECTION          ")
        log_write("==========================================================================")

        # General Overview
        log_write("\n1. MODEL DIMENSION & STRUCTURE (With Min Up/Down Time)")
        log_write(f"  -> Problem Class:                Mixed-Integer Linear Program (MILP / CQM)")
        log_write(f"  -> Total Decision Variables:     {len(cqm.variables)}")
        log_write(f"      * Binary Qubits (u_{{i,t}}):    {len(binary_vars)} (4 generators * 24 hours)")
        log_write(f"      * Continuous MW (P_{{i,t}}):     {len(real_vars)} (4 generators * 24 hours)")
        log_write(f"  -> Total Constraints:            {num_constraints}")
        log_write(f"      * Coupling Limits (P_min/max): {4 * 24 * 2} constraints")
        log_write(f"      * Power Balance (Scenarios):   {3 * 24} constraints (3 scenarios * 24 hours)")

        # Generator Parameters
        log_write("\n2. THERMAL GENERATOR PARAMETERS")
        log_write(f"  {'Gen Name':<15} | {'Fixed Cost ($/hr)':<18} | {'Var Cost ($/MWh)':<16} | {'Min (MW)':<8} | {'Max (MW)':<8}")
        log_write("  " + "-" * 75)
        for i, gen in enumerate(problem.gen_names):
            log_write(f"  {gen:<15} | ${problem.c_fixed[i]:<17.2f} | ${problem.c_var[i]:<15.2f} | {problem.p_min[i]:<8.1f} | {problem.p_max[i]:<8.1f}")

        # Stochastic Scenario Profiles Sample
        log_write("\n3. STOCHASTIC SCENARIO PROFILES SAMPLE (Demand & Wind Power)")
        log_write(f"  {'Hour':<6} | {'Scen 1 Dem (MW)':<16} | {'Scen 1 Wind (MW)':<17} | {'Scen 1 Net Dem (MW)':<20}")
        log_write("  " + "-" * 68)
        sample_hours = [0, 6, 12, 18, 23]
        for t in sample_hours:
            dem = problem.demand[0, t]
            wnd = problem.wind[0, t]
            net_dem = dem - wnd
            log_write(f"  {t:<6} | {dem:<16.2f} | {wnd:<17.2f} | {net_dem:<20.2f}")

        # Sample Variable Names
        log_write("\n4. SAMPLE DECISION VARIABLES (Registered in CQM)")
        log_write("  -> Binary Qubit Variables (Stage 1):")
        log_write(f"       {binary_vars[:4]}")
        log_write("  -> Continuous Dispatch Variables (Stage 2):")
        log_write(f"       {real_vars[:4]}")

        # Complete Master Equation (Objective Function)
        log_write("\n5. MASTER OBJECTIVE FUNCTION (Total Production & Startup Cost)")
        log_write(f"   {cqm.objective}")

        # Complete List of All Constraints
        log_write(f"\n6. ALL REGISTERED CONSTRAINTS ({num_constraints} Total)")
        log_write("  " + "-" * 75)
        for label, constraint in cqm.constraints.items():
            log_write(f"  [Label]: {label}")
            log_write(f"  [Rule]:  {constraint.to_polystring()}")
            log_write("  " + "-" * 40)

        log_write("\n==========================================================================")
        log_write("                MODEL BLUEPRINT CREATED SUCCESSFULLY                      ")
        log_write("==========================================================================")

    # 2. Print Clean Overview to Terminal (No Truncation)
    print("==========================================================================")
    print("         STOCHASTIC 4-GENERATOR UNIT COMMITMENT MODEL INSPECTION          ")
    print("==========================================================================")
    print(f"\n[INFO] Complete details and all {num_constraints} constraints written successfully!")
    print(f"       Log file location: {os.path.abspath(log_path)}")
    print("\n--- MODEL OVERVIEW SUMMARY ---")
    print(f"  -> Problem Class:               Mixed-Integer Linear Program (MILP / CQM)")
    print(f"  -> Total Decision Variables:    {len(cqm.variables)} ({len(binary_vars)} binary, {len(real_vars)} continuous)")
    print(f"  -> Total Constraints Logged:    {num_constraints}")
    print("\==========================================================================")
    print("                CONSOLE SUMMARY COMPLETED SUCCESSFULLY                    ")
    print("==========================================================================")