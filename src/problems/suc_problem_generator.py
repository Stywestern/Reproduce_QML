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
            shed_cost=actual_shed_cost
        )

# ###########################################################################################################################################
# 2. Mathematical Backend - Enables transfering of the problem into any preprocessing algorithm
# ###########################################################################################################################################
@dataclass
class Variable:
    name: str           # gen1_s3_t12
    vtype: str          # "B", "I", "C"
    lower: float        # Lowest cap of an initiated variable's value
    upper: float        # Highest

@dataclass
class Constraint:
    coefficients: dict[str, float] # like "P_gen0_hr0_scen0": 1.0
    sense: str    # "<=", ">=", "="
    rhs: float
    name: str

@dataclass
class Objective:
    coefficients: dict[str, float]
    constant: float = 0.0
    sense: str = "min"

@dataclass
class MathematicalModel:
    variables: list[Variable]
    constraints: list[Constraint]
    objective: Objective

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
        self._variables: list[Variable] = []
        self._constraints: list[Constraint] = []
        self._objective = Objective(coefficients={})
        self._index: dict[str, int] = {}

    def add_var(self, name, vtype, lower=0.0, upper=1.0):
        assert name not in self._index, f"duplicate variable {name}"
        assert vtype in ("B", "I", "C"), f"unknown variable type {vtype!r}"

        idx = len(self._variables)
        self._index[name] = idx
        self._variables.append(Variable(name, vtype, lower, upper))

    def add_objective_term(self, name, coeff):
        idx = self._index[name]
        self._objective.coefficients[idx] = \
            self._objective.coefficients.get(idx, 0.0) + coeff

    def add_constraint(self, coeffs, rhs, sense="<=", label=""):
        assert sense in {"<=", ">=", "="}

        self._constraints.append(
            Constraint(
                coefficients={self._index[name]: coeff for name, coeff in coeffs.items()},
                sense=sense,
                rhs=rhs,
                name=label
            )
        )

    def build(self) -> MathematicalModel:
        return MathematicalModel(
            variables=self._variables,
            constraints=self._constraints,
            objective=self._objective
        )

    def compile(self) -> MixedLBOProblem:
        model = self.build()

        n = len(model.variables)
        m = len(model.constraints)

        c = np.zeros(n)
        for i, coeff in model.objective.coefficients.items():
            c[i] = coeff

        row_idx, col_idx, data = [], [], []
        row_lower = np.empty(m)
        row_upper = np.empty(m)

        for r, constraint in enumerate(model.constraints):
            for i, coeff in constraint.coefficients.items():
                if coeff != 0:
                    row_idx.append(r)
                    col_idx.append(i)
                    data.append(coeff)

            if constraint.sense == "<=":
                row_lower[r], row_upper[r] = -np.inf, constraint.rhs
            elif constraint.sense == ">=":
                row_lower[r], row_upper[r] = constraint.rhs, np.inf
            else:
                row_lower[r] = row_upper[r] = constraint.rhs

        A = sp.coo_matrix(
            (data, (row_idx, col_idx)),
            shape=(m, n)
        ).tocsr()

        return MixedLBOProblem(
            var_names=[v.name for v in model.variables],
            var_types=[v.vtype for v in model.variables],
            lower=np.array([v.lower for v in model.variables]),
            upper=np.array([v.upper for v in model.variables]),
            c=c,
            A=A,
            row_lower=row_lower,
            row_upper=row_upper,
            constraint_labels=[c.name for c in model.constraints]
        )

# ###########################################################################################################################################
# 3. Model Formulation - creates the objective function and basic constraints
# ###########################################################################################################################################

class SUCModelBuilder:
    """
    Builds the baseline Stochastic Unit Commitment (SUC) Problem.
    It ONLY defines the variables, the objective function, and the fundamental 
    laws of physics (power balance & generator capacity limits).
    
    The self.builder remains open so custom constraints can be added later!
    """
    def __init__(self, problem_data: SUCProblemData):
        self.pd = problem_data
        self.builder = LBOBuilder()

    # ----------------------------------------------------------------------
    # Helper Methods: Creating standardized names for variables
    # ----------------------------------------------------------------------
    def _u_name(self, g: int, t: int) -> str: # on/off
        return f"u_gen{g}_hr{t}"

    def _p_name(self, g: int, t: int, s: int) -> str: # MW per gen
        return f"P_gen{g}_hr{t}_scen{s}"

    def _shed_name(self, t: int, s: int) -> str: # MW deficit
        return f"shed_hr{t}_scen{s}"

    # ----------------------------------------------------------------------
    # Step 1: Declare all the variables
    # ----------------------------------------------------------------------
    def setup_variables(self):
        N, T, S = self.pd.num_gens, self.pd.num_hours, self.pd.num_scenarios
        
        # 1. Binary Commitment Variables (u)
        for g in range(N):
            for t in range(T):
                self.builder.add_var(self._u_name(g, t), "B", lower=0.0, upper=1.0) # 'B' means Binary (0 or 1)

        # 2. Continuous Power (P) and Load Shedding (shed) variables
        for s in range(S):
            for t in range(T):
                self.builder.add_var(self._shed_name(t, s), "C", lower=0.0, upper=10000.0) # Shedding can go from 0 up to a huge number depending on problem (10,000 as a safe upper bound)
                for g in range(N):
                    self.builder.add_var(self._p_name(g, t, s), "C", lower=0.0, upper=self.pd.p_max[g]) # Power 'P' is Continuous ('C'). It can be 0 or go up to the generator's max.

    # ----------------------------------------------------------------------
    # Step 2: Formulate the Objective Function
    # ----------------------------------------------------------------------
    def setup_objective(self):
        N, T, S = self.pd.num_gens, self.pd.num_hours, self.pd.num_scenarios
        
        # 1. Fixed Commitment Costs
        for g in range(N):
            for t in range(T):
                self.builder.add_objective_term(self._u_name(g, t), self.pd.c_fixed[g])

        # 2. Expected Dispatch Costs
        for s in range(S):
            prob = self.pd.scenario_probs[s] # The probability p_omega
            for t in range(T):
                expected_shed_cost = prob * self.pd.shed_cost
                self.builder.add_objective_term(self._shed_name(t, s), expected_shed_cost)
                for g in range(N):
                    expected_var_cost = prob * self.pd.c_var[g]
                    self.builder.add_objective_term(self._p_name(g, t, s), expected_var_cost)

    # ----------------------------------------------------------------------
    # Step 3: Formulate the Core Constraints
    # ----------------------------------------------------------------------

    def setup_power_balance_constraints(self):
        N, T, S = self.pd.num_gens, self.pd.num_hours, self.pd.num_scenarios

        for s in range(S):
            for t in range(T):
                net_demand = self.pd.demand[s, t] - self.pd.wind[s, t]
                coeffs = {self._shed_name(t, s): 1.0}
                for g in range(N):
                    coeffs[self._p_name(g, t, s)] = 1.0
                self.builder.add_constraint(coeffs, rhs=net_demand, sense="=", label=f"balance_s{s}_t{t}")

    
    def setup_generator_limit_constraints(self):
        N, T, S = self.pd.num_gens, self.pd.num_hours, self.pd.num_scenarios
        for s in range(S):
            for t in range(T):
                for g in range(N):
                    u, p = self._u_name(g, t), self._p_name(g, t, s)
                    self.builder.add_constraint(
                        {p: 1.0, u: -self.pd.p_max[g]}, rhs=0.0, sense="<=",
                        label=f"upper_s{s}_t{t}_g{g}"
                    )
                    self.builder.add_constraint(
                        {p: -1.0, u: self.pd.p_min[g]}, rhs=0.0, sense="<=",
                        label=f"lower_s{s}_t{t}_g{g}"
                    )

    def setup_min_up_down_constraints(self):
        N, T = self.pd.num_gens, self.pd.num_hours
        for g in range(N):
            for t in range(T):
                u = self._u_name(g, t)
                prev = self.pd.initial_status[g] if t == 0 else self._u_name(g, t - 1)
                const = t == 0

                # Min Up: u_t - u_{t-1} - u_tau <= 0
                for tau in range(t + 1, min(t + self.pd.min_up[g], T)):
                    ut = self._u_name(g, tau)
                    coeffs = {u: 1.0, ut: -1.0}
                    if not const:
                        coeffs[prev] = -1.0
                    
                    self.builder.add_constraint(
                        coeffs, 
                        rhs=float(prev) if const else 0.0, 
                        sense="<=",
                        label=f"min_up_g{g}_t{t}_tau{tau}"
                    )

                # Min Down: u_{t-1} - u_t + u_tau <= 1
                for tau in range(t + 1, min(t + self.pd.min_down[g], T)):
                    ut = self._u_name(g, tau)
                    coeffs = {u: -1.0, ut: 1.0}
                    if not const:
                        coeffs[prev] = 1.0
                        
                    self.builder.add_constraint(
                        coeffs, 
                        rhs=1.0 - float(prev) if const else 1.0, 
                        sense="<=",
                        label=f"min_down_g{g}_t{t}_tau{tau}"
                    )

    def setup_core_constraints(self):
        self.setup_power_balance_constraints()
        self.setup_generator_limit_constraints()
        self.setup_min_up_down_constraints()
        
    # ----------------------------------------------------------------------
    # Execution: Compile it into the Universal Math Object
    # ----------------------------------------------------------------------
    def get_base_model(self) -> MixedLBOProblem:
        """Runs the setup and returns the compiled mathematical matrices."""
        self.setup_variables()
        self.setup_objective()
        self.setup_core_constraints()
        return self.builder.build()


############################################################################################
#                                   Execution Block
############################################################################################
if __name__ == "__main__":
    import os

    pd = SUCProblemData.generate(
        num_gens=4,
        num_hours=24,
        num_scenarios=3,
        seed=2
    )

    builder = SUCModelBuilder(pd)

    # Build the ground mathematical model
    model = builder.get_base_model()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_path = os.path.join(script_dir, "4gen_suc_problem.txt")

    binary_vars = [v for v in model.variables if v.vtype == "B"]
    continuous_vars = [v for v in model.variables if v.vtype == "C"]

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
        write(f"  Total Variables:           {len(model.variables)}")
        write(f"    Binary:                  {len(binary_vars)}")
        write(f"    Continuous:              {len(continuous_vars)}")
        write(f"  Total Constraints:         {len(model.constraints)}")

        write("\n2. GENERATOR PARAMETERS")
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

        write("\n3. SAMPLE SCENARIO DATA")
        write(f"  {'Hour':<8} {'Demand':<12} {'Wind':<12} {'Net Demand':<12}")

        for t in [0, 6, 12, 18, 23]:
            write(
                f"  {t:<8} "
                f"{pd.demand[0,t]:<12.2f} "
                f"{pd.wind[0,t]:<12.2f} "
                f"{pd.demand[0,t] - pd.wind[0,t]:<12.2f}"
            )

        write("\n4. SAMPLE VARIABLES")

        write("  Binary:")
        for v in binary_vars[:6]:
            write(f"    {v.name}: {v.vtype} [{v.lower}, {v.upper}]")

        write("  Continuous:")
        for v in continuous_vars[:6]:
            write(f"    {v.name}: {v.vtype} [{v.lower}, {v.upper}]")

        write("\n5. OBJECTIVE FUNCTION")

        write(f"  Sense: {model.objective.sense}")
        write(f"  Constant: {model.objective.constant}")
        write(f"  Number of objective terms: "
              f"{len(model.objective.coefficients)}")

        for idx, coeff in list(model.objective.coefficients.items())[:15]:
            write(f"    {coeff:+.3f} * {model.variables[idx].name}")

        write("\n6. CONSTRAINTS")

        for constraint in model.constraints:
            terms = " ".join(
                f"{coef:+g}*{model.variables[idx].name}"
                for idx, coef in constraint.coefficients.items()
            )

            write(
                f"  [{constraint.name}] "
                f"{terms} {constraint.sense} {constraint.rhs}"
            )

        write("\n" + "=" * 75)
        write("                 MODEL BLUEPRINT CREATED")
        write("=" * 75)

    print("=" * 75)
    print("       STOCHASTIC 4-GENERATOR UNIT COMMITMENT MODEL INSPECTION")
    print("=" * 75)
    print(f"\nVariables:    {len(model.variables)}")
    print(f"  Binary:     {len(binary_vars)}")
    print(f"  Continuous: {len(continuous_vars)}")
    print(f"Constraints:  {len(model.constraints)}")
    print(f"\nLog written to:")
    print(os.path.abspath(log_path))