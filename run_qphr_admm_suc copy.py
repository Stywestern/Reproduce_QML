"""
Classical Benders master problem for the SUC MILP, eqs. (2a)-(2f) of
Hong, Xu & Teng (arXiv:2502.15917). Solved with CBC for now.

Style: no function definitions except storage classes. Every step runs
top to bottom, in the order the paper presents it.

Structure worth noting: the master is assembled as its own MILPModel
FIRST (reusing MILPBuilder from suc_model.py), and only the very last
section translates that into pulp and solves it. That split is
deliberate -- when PHR-ALM replaces CBC with a QUBO/QPU solve, the
assembly section above stays untouched; only the final translate+solve
block gets replaced with one that reads the same MILPModel (its rows
are already in exactly the g_i(x) <= 0 form eq. (10) needs) and builds
a penalty-augmented QUBO instead of pulp constraints.
"""

from __future__ import annotations
from dataclasses import dataclass
import pulp
import numpy as np

from src.problems.suc_problem_generator import SUCProblemData, SUCModelBuilder, MILPModel, MILPBuilder

@dataclass
class BendersCut:
    """
    One aggregated multi-scenario optimality cut, eq. (2c), rearranged into

        Upsilon - sum_g,t theta[u_name] * u[u_name]  >=  rhs

    theta and rhs are already the SUM over scenarios h (each subproblem's
    dual/cost is pi(xi_h)-weighted internally) -- one aggregated cut per
    BD iteration, not one cut per scenario.
    """
    theta: dict[str, float]   # u-variable name -> aggregated dual
    rhs: float                # aggregate_subproblem_cost - sum(theta * u_l)
    iteration: int            # which BD iteration l produced this cut



# ============================================================================
# Build the SUC problem once -- shared by every master/subproblem below.
# ============================================================================
 
pd = SUCProblemData.generate(num_gens=4, num_hours=24, num_scenarios=3, seed=2)
model = SUCModelBuilder(pd).get_base_model()
 
first_stage_names = set(model.var_names[i] for i in model.first_stage_vars())
u_fixed_prefix = "ufix__"
 
# ============================================================================
# Benders loop
# ============================================================================
 
cuts: list[BendersCut] = []
upsilon_lower = 0.0
max_iterations = 20
gap_tolerance = 1e-4
 
lower_bound = -np.inf
upper_bound = np.inf
best_u = None
iteration = 0
 
while iteration < max_iterations and (upper_bound - lower_bound) > gap_tolerance * max(1.0, abs(upper_bound)):
    # ------------------------------------------------------------------
    # Master: assemble as its own MILPModel (unchanged from step 1)
    # ------------------------------------------------------------------
    master_builder = MILPBuilder()
 
    for i in model.first_stage_vars():
        name = model.var_names[i]
        master_builder.add_var(name, "B", 0.0, 1.0)
        if model.c[i] != 0:
            master_builder.add_obj(name, float(model.c[i]))
 
    master_builder.add_var("Upsilon", "C", upsilon_lower, np.inf)
    master_builder.add_obj("Upsilon", 1.0)
 
    for r in model.shared_rows():
        row = model.A.getrow(r)
        terms = {model.var_names[j]: float(v) for j, v in zip(row.indices, row.data)}
        sense, rhs = model.row_sense_rhs(r)
        master_builder.add_row(terms, sense, rhs, model.constraint_labels[r])
 
    for cut in cuts:
        terms = {name: -theta for name, theta in cut.theta.items()}
        terms["Upsilon"] = 1.0
        master_builder.add_row(terms, ">=", cut.rhs, f"benders_cut_l{cut.iteration}")
 
    master_model = master_builder.build()
 
    # ------------------------------------------------------------------
    # Master: translate + solve -- the pulp-specific section that step 2
    # replaces with a QUBO build + sampler call, reading the same
    # master_model produced above.
    # ------------------------------------------------------------------
    master_prob = pulp.LpProblem(f"Benders_Master_k{iteration}", pulp.LpMinimize)
 
    master_pulp_vars: dict[str, pulp.LpVariable] = {}
    for i, name in enumerate(master_model.var_names):
        vtype = pulp.LpBinary if master_model.var_types[i] == "B" else pulp.LpContinuous
        lb = None if master_model.lower[i] == -np.inf else float(master_model.lower[i])
        ub = None if master_model.upper[i] == np.inf else float(master_model.upper[i])
        master_pulp_vars[name] = pulp.LpVariable(name, lowBound=lb, upBound=ub, cat=vtype)
 
    master_prob += pulp.lpSum(
        float(master_model.c[i]) * master_pulp_vars[master_model.var_names[i]]
        for i in range(len(master_model.c))
        if master_model.c[i] != 0
    )
 
    for r in range(master_model.A.shape[0]):
        row = master_model.A.getrow(r)
        expr = pulp.lpSum(
            float(v) * master_pulp_vars[master_model.var_names[j]]
            for j, v in zip(row.indices, row.data)
        )
        sense, rhs = master_model.row_sense_rhs(r)
        label = master_model.constraint_labels[r]
        if sense == "=":
            master_prob += (expr == rhs, label)
        elif sense == "<=":
            master_prob += (expr <= rhs, label)
        else:
            master_prob += (expr >= rhs, label)
 
    master_status_code = master_prob.solve(pulp.PULP_CBC_CMD(msg=False))
    master_status = pulp.LpStatus[master_status_code]
 
    u_values = {
        name: int(round(var.varValue))
        for name, var in master_pulp_vars.items()
        if name != "Upsilon"
    }
    fixed_cost = sum(
        float(model.c[i]) * u_values[model.var_names[i]]
        for i in model.first_stage_vars()
        if model.c[i] != 0
    )
    if master_status == "Optimal":
        lower_bound = float(pulp.value(master_prob.objective))
 
    # ------------------------------------------------------------------
    # Subproblems: one per scenario, using THIS iteration's u_values.
    # Same assembly pattern as the single-scenario test, run K times.
    # ------------------------------------------------------------------
    total_dispatch_cost = 0.0
    theta_agg = {name: 0.0 for name in first_stage_names}
 
    for scenario in range(pd.num_scenarios):
 
        sub_builder = MILPBuilder()
 
        for i in model.scenario_vars(scenario):
            name = model.var_names[i]
            sub_builder.add_var(name, "C", float(model.lower[i]), float(model.upper[i]))
            if model.c[i] != 0:
                sub_builder.add_obj(name, float(model.c[i]))
 
        for i in model.first_stage_vars():
            name = model.var_names[i]
            sub_builder.add_var(u_fixed_prefix + name, "C", 0.0, 1.0)
 
        for r in model.scenario_rows(scenario):
            row = model.A.getrow(r)
            terms = {}
            for j, v in zip(row.indices, row.data):
                name = model.var_names[j]
                if name in first_stage_names:
                    name = u_fixed_prefix + name
                terms[name] = float(v)
            sense, rhs = model.row_sense_rhs(r)
            sub_builder.add_row(terms, sense, rhs, model.constraint_labels[r])
 
        for i in model.first_stage_vars():
            name = model.var_names[i]
            sub_builder.add_row({u_fixed_prefix + name: 1.0}, "=", float(u_values[name]), f"fix_{name}")
 
        sub_model = sub_builder.build()
 
        sub_prob = pulp.LpProblem(f"Benders_Subproblem_s{scenario}_k{iteration}", pulp.LpMinimize)
 
        sub_pulp_vars: dict[str, pulp.LpVariable] = {}
        for i, name in enumerate(sub_model.var_names):
            lb = None if sub_model.lower[i] == -np.inf else float(sub_model.lower[i])
            ub = None if sub_model.upper[i] == np.inf else float(sub_model.upper[i])
            sub_pulp_vars[name] = pulp.LpVariable(name, lowBound=lb, upBound=ub, cat=pulp.LpContinuous)
 
        sub_prob += pulp.lpSum(
            float(sub_model.c[i]) * sub_pulp_vars[sub_model.var_names[i]]
            for i in range(len(sub_model.c))
            if sub_model.c[i] != 0
        )
 
        sub_pulp_constraints: dict[str, pulp.LpConstraint] = {}
        for r in range(sub_model.A.shape[0]):
            row = sub_model.A.getrow(r)
            expr = pulp.lpSum(
                float(v) * sub_pulp_vars[sub_model.var_names[j]]
                for j, v in zip(row.indices, row.data)
            )
            sense, rhs = sub_model.row_sense_rhs(r)
            label = sub_model.constraint_labels[r]
            if sense == "=":
                sub_prob += (expr == rhs, label)
            elif sense == "<=":
                sub_prob += (expr <= rhs, label)
            else:
                sub_prob += (expr >= rhs, label)
            sub_pulp_constraints[label] = sub_prob.constraints[label]
 
        sub_status_code = sub_prob.solve(pulp.PULP_CBC_CMD(msg=False))
        sub_status = pulp.LpStatus[sub_status_code]
        sub_cost = float(pulp.value(sub_prob.objective)) if sub_status == "Optimal" else None
 
        if sub_status != "Optimal":
            print(f"WARNING: scenario {scenario} subproblem status = {sub_status} at iteration {iteration}")
            sub_cost = 0.0  # placeholder -- infeasibility cuts are not handled yet
 
        total_dispatch_cost += sub_cost
 
        for i in model.first_stage_vars():
            name = model.var_names[i]
            theta_agg[name] += sub_pulp_constraints[f"fix_{name}"].pi
 
    # ------------------------------------------------------------------
    # Aggregate into one cut (2c), update LB/UB, log, advance
    # ------------------------------------------------------------------
    rhs = total_dispatch_cost - sum(theta_agg[name] * u_values[name] for name in theta_agg)
    cuts.append(BendersCut(theta=dict(theta_agg), rhs=rhs, iteration=iteration))
 
    candidate_upper_bound = fixed_cost + total_dispatch_cost
    if candidate_upper_bound < upper_bound:
        upper_bound = candidate_upper_bound
        best_u = dict(u_values)
 
    gap = upper_bound - lower_bound
    committed = sum(u_values.values())
    print(f"Iteration {iteration}: LB={lower_bound:.2f}  UB={upper_bound:.2f}  gap={gap:.2f}  committed={committed}")
 
    iteration += 1
 
print(f"\nFinished after {iteration} iteration(s). Final LB={lower_bound:.2f}  UB={upper_bound:.2f}")
print(f"Best committed count: {sum(best_u.values()) if best_u else 'n/a'}")