############################################################################################
#                      PROBLEM: 4-GENERATOR STOCHASTIC UC MODEL
############################################################################################

# Third Party Libraries
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import scipy.sparse as sp

# ###########################################################################################################################################
# 1. Problem data -- fully parameterized so S/T/K can all be changed. One can even add more parameters with some work.
# ###########################################################################################################################################

@dataclass
class SUCProblemData:
    num_gens: int              # N
    num_hours: int             # T
    num_scenarios: int         # S
    p_min: np.ndarray          # (N,), p = power of generator in MW
    p_max: np.ndarray          # (N,)
    c_var: np.ndarray          # (N,)  $/MWh, c = cost of generators variable output
    c_fixed: np.ndarray        # (N,)  $/h, cost of just running it on
    min_up: np.ndarray         # (N,) integer hours, T^U_g, a generator must stay on for this much hours when opened
    min_down: np.ndarray       # (N,) integer hours, T^D_g
    initial_status: np.ndarray  # (N,), 0 = initially off, 1 = initially on
    scenario_probs: np.ndarray  # (S,), sums to 1
    demand: np.ndarray         # (S, T), generators must met this
    wind: np.ndarray           # (S, T), eases the generator load
    shed_cost: float = 1000.0  # $/MWh, If in any scenario we have to cut electricity from people, we want to penalize that
    spill_cost: float = 100.0    # $/MWh, Cost/penalty for curtailing excess generation (spill), we also want to penalize that but not that much, it can be redirected

    def __post_init__(self):
            N, T, S = self.num_gens, self.num_hours, self.num_scenarios
            assert self.p_min.shape == (N,) and self.p_max.shape == (N,)
            assert np.all(self.p_min <= self.p_max), "p_min must be <= p_max for every generator"
            assert self.c_var.shape == (N,) and self.c_fixed.shape == (N,)
            assert self.min_up.shape == (N,) and self.min_down.shape == (N,)
            assert np.all(self.min_up >= 0) and np.all(self.min_down >= 0)
            assert self.initial_status.shape == (N,)
            assert np.all(np.isin(self.initial_status, [0, 1]))
            assert self.scenario_probs.shape == (S,)
            assert abs(self.scenario_probs.sum() - 1.0) < 1e-9, "scenario_probs must sum to 1"
            assert self.demand.shape == (S, T) and self.wind.shape == (S, T)

    @classmethod
    def generate(cls, num_gens: int, num_hours: int, num_scenarios: int,seed: int = 2,
                 p_min: np.ndarray | None = None,
                 p_max: np.ndarray | None = None,
                 c_var: np.ndarray | None = None,
                 c_fixed: np.ndarray | None = None,
                 min_up: np.ndarray | None = None,
                 min_down: np.ndarray | None = None,
                 initial_status: np.ndarray | None = None,
                 scenario_probs: np.ndarray | None = None,
                 demand: np.ndarray | None = None,
                 wind: np.ndarray | None = None,
                 shed_cost: float | None = None,
                 spill_cost: float | None = None) -> "SUCProblemData":
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
        actual_spill_cost = spill_cost if spill_cost is not None else 100.0

        # 3. Operational Structures
        if min_up is not None:
            actual_min_up = min_up
        else:
            actual_min_up = np.clip((6 - 4 * tier).round().astype(int), 1, None)
            
        if min_down is not None:
            actual_min_down = min_down
        else:
            actual_min_down = actual_min_up.copy()

        if initial_status is not None:
            actual_initial_status = initial_status
        else:
            actual_initial_status = np.zeros(num_gens, dtype=int)

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
            initial_status=actual_initial_status,
            scenario_probs=actual_probs, 
            demand=actual_demand, 
            wind=actual_wind,
            shed_cost=actual_shed_cost,
            spill_cost=actual_spill_cost
        )

# ============================================================================
# 2. Lean LP/MILP backend.
#
#    ONE model, ONE builder. No Variable/Constraint/Objective object graph
#    sitting in between -- rows and objective terms are staged directly as
#    (index, value) triplets and assembled into a CSR matrix once, at
#    build() time. This is the only representation downstream code sees.
# ============================================================================
 
@dataclass
class MILPModel:
    """
    Mixed-Integer Linear Program:
        min c^T x   s.t.  row_lower <= A x <= row_upper,   lower <= x <= upper
        x_i binary if var_types[i] == 'B', continuous if 'C'.
    The 'B' columns are what make this MILP rather than LP -- CBC needs
    branch-and-cut (not a pure simplex/interior-point solve) because of them,
    and they're exactly what a Benders master problem isolates.
 
    var_scenario / row_scenario tag which scenario a variable or row
    belongs to (-1 = first-stage / shared across scenarios). This is the
    seam Benders and ADMM/progressive-hedging decomposition split along,
    so it's recorded once here instead of re-derived later by parsing
    variable-name strings.
    """

    var_names: list[str]
    var_types: np.ndarray        # 'B' / 'C' per variable
    lower: np.ndarray            # l_var
    upper: np.ndarray            # u_var
    c: np.ndarray                # Objective cost vector c
    A: sp.csr_matrix             # Sparse constraint coefficient matrix A
    row_lower: np.ndarray        # l_row
    row_upper: np.ndarray        # u_row
    constraint_labels: list[str]
    var_scenario: np.ndarray     # Scenario tagging vector (-1 = Stage 1, 0..K-1 = Stage 2)
    row_scenario: np.ndarray     # Row scenario tagging vector
    _index: dict = None
 
    def var_index(self, name: str) -> int:
        return self._index[name]
 
    def first_stage_vars(self) -> np.ndarray:
        return np.where(self.var_scenario == -1)[0]
 
    def scenario_vars(self, s: int) -> np.ndarray:
        return np.where(self.var_scenario == s)[0]
 
    def shared_rows(self) -> np.ndarray:
        return np.where(self.row_scenario == -1)[0]
 
    def scenario_rows(self, s: int) -> np.ndarray:
        return np.where(self.row_scenario == s)[0]
 
    def row_sense_rhs(self, r: int) -> tuple[str, float]:
        """Reconstructs ('<=' | '>=' | '=', rhs) for row r from its bounds."""
        lo, hi = self.row_lower[r], self.row_upper[r]
        if lo == hi:
            return "=", lo
        if lo == -np.inf:
            return "<=", hi
        return ">=", lo

    def add_row(self, coeffs: dict[str, float], sense: str, rhs: float, label: str = "", scenario: int | None = None):
        """
        Incrementally appends a single constraint row directly to the built matrices.
        """
        assert sense in ("<=", ">=", "="), f"unknown sense {sense!r}"
        
        # 1. Parse coefficients into column indices
        row_cols = []
        row_vals = []
        for name, coeff in coeffs.items():
            if coeff != 0:
                row_cols.append(self._index[name])
                row_vals.append(coeff)
                
        # 2. Build a 1-row sparse matrix for the new constraint
        n_cols = len(self.var_names)
        new_row_matrix = sp.coo_matrix(
            (row_vals, ([0] * len(row_cols), row_cols)), 
            shape=(1, n_cols)
        ).tocsr()
        
        # 3. Vertically stack it onto the existing A matrix
        self.A = sp.vstack([self.A, new_row_matrix]).tocsr()
        
        # 4. Calculate bounds
        lo, hi = {"<=": (-np.inf, rhs), ">=": (rhs, np.inf), "=": (rhs, rhs)}[sense]
        
        # 5. Append to the numpy arrays and lists
        self.row_lower = np.append(self.row_lower, lo)
        self.row_upper = np.append(self.row_upper, hi)
        self.row_scenario = np.append(self.row_scenario, -1 if scenario is None else scenario)
        self.constraint_labels.append(label)
 
 
class MILPBuilder:
    """
    add_var(name, vtype, lower, upper, scenario=None)
    add_obj(name, coeff)                                    -- accumulates
    add_row({name: coeff, ...}, sense, rhs, scenario=None)  -- one row
    build() -> MILPModel                                       -- single assembly pass
 
    scenario=None marks a variable/row as first-stage / shared; pass the
    scenario index for anything that only exists within one scenario's
    subproblem.
    """
    def __init__(self):
        # Variable definitions (columns)
        self._names: list[str] = []          # String identifiers for each variable
        self._types: list[str] = []          # Variable type: 'B' (Binary) or 'C' (Continuous)
        self._lower: list[float] = []        # Lower bound limits for variables
        self._upper: list[float] = []        
        self._var_scenario: list[int] = []   # -1 for first-stage/shared variables, or scenario integer index
        self._index: dict[str, int] = {}     # Maps variable string name to its column integer index
        
        # Objective function
        self._obj: dict[int, float] = {}     # Maps variable column index to its objective function coefficient

        # Constraint matrix (Coordinate List / COO sparse matrix format), like row_ij = v
        self._rows_i: list[int] = []         # Row indices for non-zero constraint coefficients
        self._rows_j: list[int] = []         # Column indices (variable IDs) for non-zero constraint coefficients
        self._rows_v: list[float] = []       # The actual non-zero coefficient values (multiplier)

        # Constraint definitions (rows)
        self._row_lower: list[float] = []    # Constraint lower bounds (RHS for >= and ==; -inf for <=)
        self._row_upper: list[float] = []    
        self._row_labels: list[str] = []     # String identifiers for each constraint row
        self._row_scenario: list[int] = []   # -1 for shared constraints, or scenario integer index
 
    def add_var(self, name: str, vtype: str, lower: float = 0.0, upper: float = 1.0,
                scenario: int | None = None):
        assert name not in self._index, f"duplicate variable {name}"
        assert vtype in ("B", "C"), f"unknown variable type {vtype!r}"
        self._index[name] = len(self._names)
        self._names.append(name)
        self._types.append(vtype)
        self._lower.append(lower)
        self._upper.append(upper)
        self._var_scenario.append(-1 if scenario is None else scenario)
 
    def add_obj(self, name: str, coeff: float):
        i = self._index[name]
        self._obj[i] = self._obj.get(i, 0.0) + coeff
 
    def add_row(self, coeffs: dict[str, float], sense: str, rhs: float, label: str = "",
                scenario: int | None = None):
        assert sense in ("<=", ">=", "="), f"unknown sense {sense!r}"
        r = len(self._row_labels)
        self._row_scenario.append(-1 if scenario is None else scenario)
        for name, coeff in coeffs.items():
            if coeff != 0:
                self._rows_i.append(r)
                self._rows_j.append(self._index[name])
                self._rows_v.append(coeff)
        lo, hi = {"<=": (-np.inf, rhs), ">=": (rhs, np.inf), "=": (rhs, rhs)}[sense]
        self._row_lower.append(lo)
        self._row_upper.append(hi)
        self._row_labels.append(label)
 
    def build(self) -> MILPModel:
        n, m = len(self._names), len(self._row_labels)
        c = np.zeros(n)
        for i, v in self._obj.items():
            c[i] = v
        A = sp.coo_matrix(
            (self._rows_v, (self._rows_i, self._rows_j)), shape=(m, n)
        ).tocsr()
        return MILPModel(
            var_names=self._names,
            var_types=np.array(self._types),
            lower=np.array(self._lower),
            upper=np.array(self._upper),
            c=c,
            A=A,
            row_lower=np.array(self._row_lower),
            row_upper=np.array(self._row_upper),
            constraint_labels=self._row_labels,
            var_scenario=np.array(self._var_scenario),
            row_scenario=np.array(self._row_scenario),
            _index=dict(self._index),
        )
 
 
# ============================================================================
# 3. Model Formulation -- variables, objective, and the core physics.
# ============================================================================
 
class SUCModelBuilder:
    """
    Builds the baseline Stochastic Unit Commitment (SUC) problem: variables,
    objective, and power balance / capacity / min up-down constraints.
    self.builder stays open afterward so custom constraints can be layered on.
    """
    def __init__(self, problem_data: SUCProblemData):
        self.pd = problem_data
        self.builder = MILPBuilder()
 
    # -- variable naming ------------------------------------------------
    def _u_name(self, g: int, t: int) -> str:
        return f"u_gen{g}_hr{t}"
 
    def _p_name(self, g: int, t: int, s: int) -> str:
        return f"P_gen{g}_hr{t}_scen{s}"
 
    def _shed_name(self, t: int, s: int) -> str:
        return f"shed_hr{t}_scen{s}"

    def _spill_name(self, t: int, s: int) -> str:
        return f"spill_hr{t}_scen{s}"
 
    # -- variables --------------------------------------------------------
    def setup_variables(self):
        N, T, S = self.pd.num_gens, self.pd.num_hours, self.pd.num_scenarios
        for g in range(N):
            for t in range(T):
                self.builder.add_var(self._u_name(g, t), "B", 0.0, 1.0)  # first-stage
        for s in range(S):
            for t in range(T):
                self.builder.add_var(self._shed_name(t, s), "C", 0.0, 10000.0, scenario=s)
                self.builder.add_var(self._spill_name(t, s), "C", 0.0, 10000.0, scenario=s)
                for g in range(N):
                    self.builder.add_var(self._p_name(g, t, s), "C", 0.0, self.pd.p_max[g],
                                          scenario=s)
 
    # -- objective ----------------------------------------------------------
    def setup_objective(self):
        N, T, S = self.pd.num_gens, self.pd.num_hours, self.pd.num_scenarios
        for g in range(N):
            for t in range(T):
                self.builder.add_obj(self._u_name(g, t), self.pd.c_fixed[g])
        for s in range(S):
            prob = self.pd.scenario_probs[s]
            for t in range(T):
                self.builder.add_obj(self._shed_name(t, s), prob * self.pd.shed_cost)
                self.builder.add_obj(self._spill_name(t, s), prob * self.pd.spill_cost)
                for g in range(N):
                    self.builder.add_obj(self._p_name(g, t, s), prob * self.pd.c_var[g])
 
    # -- core constraints -----------------------------------------------
    def setup_power_balance_constraints(self):
        N, T, S = self.pd.num_gens, self.pd.num_hours, self.pd.num_scenarios
        for s in range(S):
            for t in range(T):
                net_demand = self.pd.demand[s, t] - self.pd.wind[s, t]
                coeffs = {self._shed_name(t, s): 1.0, self._spill_name(t, s): -1.0}
                for g in range(N):
                    coeffs[self._p_name(g, t, s)] = 1.0
                self.builder.add_row(coeffs, "=", net_demand, f"balance_s{s}_t{t}", scenario=s)
 
    def setup_generator_limit_constraints(self):
        N, T, S = self.pd.num_gens, self.pd.num_hours, self.pd.num_scenarios
        for s in range(S):
            for t in range(T):
                for g in range(N):
                    u, p = self._u_name(g, t), self._p_name(g, t, s)
                    self.builder.add_row({p: 1.0, u: -self.pd.p_max[g]}, "<=", 0.0,
                                          f"upper_s{s}_t{t}_g{g}", scenario=s)
                    self.builder.add_row({p: -1.0, u: self.pd.p_min[g]}, "<=", 0.0,
                                          f"lower_s{s}_t{t}_g{g}", scenario=s)
 
    def setup_min_up_down_constraints(self):
        N, T = self.pd.num_gens, self.pd.num_hours
        for g in range(N):
            U, D = self.pd.min_up[g], self.pd.min_down[g]
            for t in range(T):
                prev = self.pd.initial_status[g] if t == 0 else self._u_name(g, t - 1)
                cur = self._u_name(g, t)
 
                if t + U <= T:
                    coeffs = {self._u_name(g, k): 1.0 for k in range(t, t + U)}
                    coeffs[cur] = coeffs.get(cur, 0.0) - U
                    if t > 0:
                        coeffs[prev] = coeffs.get(prev, 0.0) + U
                        rhs = 0.0
                    else:
                        rhs = U * self.pd.initial_status[g]
                    self.builder.add_row(coeffs, ">=", rhs, f"min_up_g{g}_t{t}")
 
                if t + D <= T:
                    coeffs = {self._u_name(g, k): -1.0 for k in range(t, t + D)}
                    coeffs[cur] = coeffs.get(cur, 0.0) + D
                    if t > 0:
                        coeffs[prev] = coeffs.get(prev, 0.0) - D
                        rhs = 0.0
                    else:
                        rhs = -D * self.pd.initial_status[g]
                    self.builder.add_row(coeffs, "<=", rhs, f"min_down_g{g}_t{t}")
 
    def setup_core_constraints(self):
        self.setup_power_balance_constraints()
        self.setup_generator_limit_constraints()
        self.setup_min_up_down_constraints()
 
    # -- entry point ----------------------------------------------------
    def get_base_model(self) -> MILPModel:
        """Runs the setup and returns the compiled LP/MILP model directly."""
        self.setup_variables()
        self.setup_objective()
        self.setup_core_constraints()
        return self.builder.build()
 
 
############################################################################################
#                                   Execution Block
############################################################################################
if __name__ == "__main__":
    import os
 
    pd = SUCProblemData.generate(num_gens=4, num_hours=24, num_scenarios=3, seed=2)
    builder = SUCModelBuilder(pd)
    model = builder.get_base_model()
 
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_path = os.path.join(script_dir, "4gen_suc_problem.txt")
 
    binary_idx = np.where(model.var_types == "B")[0]
    continuous_idx = np.where(model.var_types == "C")[0]
 
    with open(log_path, "w", encoding="utf-8") as log:
        def write(text=""):
            log.write(text + "\n")
 
        write("=" * 75)
        write("       STOCHASTIC 4-GENERATOR UNIT COMMITMENT MODEL INSPECTION")
        write("=" * 75)
 
        write("\n1. MODEL DIMENSION & STRUCTURE")
        write(f"  Problem Class:             Stochastic Unit Commitment MILP")
        write(f"  Generators:                {pd.num_gens}")
        write(f"  Hours:                     {pd.num_hours}")
        write(f"  Scenarios:                 {pd.num_scenarios}")
        write(f"  Total Variables:           {len(model.var_names)}")
        write(f"    Binary:                  {len(binary_idx)}")
        write(f"    Continuous:              {len(continuous_idx)}")
        write(f"  Total Constraints:         {model.A.shape[0]}")
 
        # -- decomposition-relevant partition: first-stage vs per-scenario --
        write("\n2. FIRST-STAGE / SCENARIO PARTITION  (for Benders / ADMM)")
        n_first = len(model.first_stage_vars())
        n_shared_rows = len(model.shared_rows())
        write(f"  First-stage variables:     {n_first}   (commitment u_g,t -- shared master decision)")
        write(f"  Shared rows:               {n_shared_rows}   (min up/down time coupling)")
        for s in range(pd.num_scenarios):
            write(f"  Scenario {s}: "
                  f"{len(model.scenario_vars(s))} vars, "
                  f"{len(model.scenario_rows(s))} rows   "
                  f"(dispatch P, shed + balance/capacity -- independent given u)")
        write("  -> Benders: master = first-stage vars + shared rows;")
        write("             subproblem_s = scenario_vars(s)/scenario_rows(s), parameterized by u.")
        write("  -> ADMM/progressive hedging: one subproblem per scenario over its own vars,")
        write("             consensus enforced on the shared first-stage u across scenarios.")
 
        write("\n3. GENERATOR PARAMETERS")
        write(f"  {'Gen':<8} {'Fixed $/h':<12} {'Var $/MWh':<12} "
              f"{'Pmin':<10} {'Pmax':<10} {'MinUp':<8} {'MinDown':<8} {'Initial':<8}")
        for g in range(pd.num_gens):
            write(
                f"  G{g:<7} "
                f"{pd.c_fixed[g]:<12.1f} "
                f"{pd.c_var[g]:<12.1f} "
                f"{pd.p_min[g]:<10.1f} "
                f"{pd.p_max[g]:<10.1f} "
                f"{pd.min_up[g]:<8} "
                f"{pd.min_down[g]:<8} "
                f"{pd.initial_status[g]:<8}"
            )
 
        write("\n4. SAMPLE SCENARIO DATA")
        write(f"  {'Hour':<8} {'Demand':<12} {'Wind':<12} {'Net Demand':<12}")
        for t in [0, 6, 12, 18, 23]:
            write(
                f"  {t:<8} "
                f"{pd.demand[0, t]:<12.2f} "
                f"{pd.wind[0, t]:<12.2f} "
                f"{pd.demand[0, t] - pd.wind[0, t]:<12.2f}"
            )
 
        write("\n5. SAMPLE VARIABLES")
        write("  Binary (first-stage):")
        for i in binary_idx[:6]:
            write(f"    {model.var_names[i]}: {model.var_types[i]} "
                  f"[{model.lower[i]}, {model.upper[i]}]")
        write("  Continuous:")
        for i in continuous_idx[:6]:
            write(f"    {model.var_names[i]}: {model.var_types[i]} "
                  f"[{model.lower[i]}, {model.upper[i]}]  scenario={model.var_scenario[i]}")
 
        write("\n6. OBJECTIVE FUNCTION")
        nz_obj = np.nonzero(model.c)[0]
        write(f"  Sense: min")
        write(f"  Number of objective terms: {len(nz_obj)}")
        for i in nz_obj[:15]:
            write(f"    {model.c[i]:+.3f} * {model.var_names[i]}")
 
        write("\n7. CONSTRAINTS")
        for r in range(model.A.shape[0]):
            row = model.A.getrow(r)
            terms = " ".join(
                f"{coef:+g}*{model.var_names[j]}"
                for j, coef in zip(row.indices, row.data)
            )
            sense, rhs = model.row_sense_rhs(r)
            write(f"  [{model.constraint_labels[r]}] {terms} {sense} {rhs}")
 
        write("\n" + "=" * 75)
        write("                 MODEL BLUEPRINT CREATED")
        write("=" * 75)
 
    print("=" * 75)
    print("       STOCHASTIC 4-GENERATOR UNIT COMMITMENT MODEL INSPECTION")
    print("=" * 75)
    print(f"\nVariables:    {len(model.var_names)}")
    print(f"  Binary:     {len(binary_idx)}  (first-stage)")
    print(f"  Continuous: {len(continuous_idx)}  (per-scenario)")
    print(f"Constraints:  {model.A.shape[0]}")
    print(f"\nLog written to:")
    print(os.path.abspath(log_path))