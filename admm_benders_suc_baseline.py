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
from tqdm import tqdm
import os
import json

import functools
import time

def print_timed(*args, **kwargs):
    """Prints arguments prefixed with the current timestamp [HH:MM:SS]."""
    timestamp = time.strftime("[%H:%M:%S]")
    print(timestamp, *args, **kwargs)

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
num_hours = 24
num_scenarios = 3
seed = 2
 
print("\n[1. Instantiating Problem Blueprint]")
rng = np.random.default_rng(seed)
t = np.arange(num_hours)

base_demand = 220 + 90 * np.sin((t - 6) * np.pi / 12)
custom_demand = np.array([
    base_demand + rng.normal(0, 10, num_hours)
    for _ in range(num_scenarios)
])

custom_wind = np.array([
    np.clip(rng.normal(20, 15, num_hours), 0, 40)
    for _ in range(num_scenarios)
])

pd = SUCProblemData.generate(
    num_gens=4,
    num_hours=num_hours,
    num_scenarios=num_scenarios,
    seed=seed,
    p_max=np.array([100.0, 95.0, 80.0, 75.0]),
    p_min=np.array([40.0, 35.0, 30.0, 25.0]),
    c_var=np.array([25.0, 26.0, 27.0, 28.0]),
    c_fixed=np.array([300.0, 290.0, 280.0, 270.0]),
    min_up=np.array([10, 8, 8, 6]),
    min_down=np.array([10, 8, 8, 6]),
    scenario_probs=np.array([0.2, 0.5, 0.3]),
    demand=custom_demand,
    wind=custom_wind
)

model = SUCModelBuilder(pd).get_base_model()
 
first_stage_names = set(model.var_names[i] for i in model.first_stage_vars())
u_fixed_prefix = "ufix__"
 
# ============================================================================
# Benders loop
# ============================================================================
 
cuts: list[BendersCut] = []
upsilon_lower = 0.0
max_iterations = 100
gap_tolerance = 1e-4
 
lower_bound = -np.inf
upper_bound = np.inf
best_u = None
iteration = 0

pbar_outer = tqdm(
    total=max_iterations,
    desc="Benders",
    unit="iter"
)

iteration_history = []
while iteration < max_iterations:
    iteration_start = time.perf_counter()
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

    print_timed("Master solving...")
    master_start = time.perf_counter()
    master_status_code = master_prob.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=30, gapRel=0.005))
    master_status = pulp.LpStatus[master_status_code]
    print_timed("Master solving complete")
    master_time = time.perf_counter() - master_start
    print_timed(
        f"Master Upsilon it{iteration}: "
        f"{master_pulp_vars['Upsilon'].varValue}"
    )
 
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
    expected_recourse_cost = 0.0
    theta_agg = {name: 0.0 for name in first_stage_names}

    subproblem_start = time.perf_counter()
    for scenario in tqdm(range(pd.num_scenarios), desc=f"Iteration {iteration} - Subproblems", unit="scenario", leave=False):
        sub_builder = MILPBuilder()
        probs = pd.scenario_probs[scenario]

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

        print_timed("Subproblem solving...")
        sub_status_code = sub_prob.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=30))
        sub_status = pulp.LpStatus[sub_status_code]
        sub_cost = float(pulp.value(sub_prob.objective)) if sub_status == "Optimal" else None
        print_timed("Subproblem solving complete")
 
        if sub_status != "Optimal":
            raise RuntimeError(
                f"Scenario {scenario} subproblem was not optimal: "
                f"{sub_status}"
            )
 
        expected_recourse_cost += probs * sub_cost
        print_timed(f"total dispatch cost it{iteration}:", expected_recourse_cost)

        for i in model.first_stage_vars():
            name = model.var_names[i]
            theta_agg[name] += probs * sub_pulp_constraints[f"fix_{name}"].pi
    
    subproblem_time = time.perf_counter() - subproblem_start

    # ------------------------------------------------------------------
    # Aggregate into one cut (2c), update LB/UB, log, advance
    # ------------------------------------------------------------------
    rhs = expected_recourse_cost - sum(theta_agg[name] * u_values[name] for name in theta_agg)
    print_timed(f"rhs it{iteration}:", rhs)

    cuts_before = len(cuts)
    cuts.append(BendersCut(theta=dict(theta_agg), rhs=rhs, iteration=iteration))
    print_timed(f"theta range it{iteration}: "
            f"[{min(theta_agg.values()):.4f}, {max(theta_agg.values()):.4f}]")
    print_timed(f"cut rhs it{iteration}: {rhs:.4f}")

    cut_value_at_current_u = (
        rhs
        + sum(theta_agg[name] * u_values[name] for name in theta_agg)
    )
    cut_tightness_error = cut_value_at_current_u - expected_recourse_cost

    candidate_upper_bound = fixed_cost + expected_recourse_cost
    if candidate_upper_bound < upper_bound:
        upper_bound = candidate_upper_bound
        best_u = dict(u_values)
    print_timed(f"upper bound it{iteration}:", upper_bound)

    # ------------------------------------------------------------------
    # Log iterations
    # ------------------------------------------------------------------
    iteration_time = time.perf_counter() - iteration_start
    iteration_history.append({
        "iteration": int(iteration),
        "master_status": master_status,
        "master_objective": (
            float(pulp.value(master_prob.objective))
            if pulp.value(master_prob.objective) is not None
            else None
        ),
        "fixed_cost": float(fixed_cost),
        "recourse_cost": float(expected_recourse_cost),
        "candidate_upper_bound": float(candidate_upper_bound),
        "global_lower_bound": (
            float(lower_bound) if np.isfinite(lower_bound) else None
        ),
        "global_upper_bound": (
            float(upper_bound) if np.isfinite(upper_bound) else None
        ),
        "absolute_gap": (
            float(upper_bound - lower_bound)
            if np.isfinite(lower_bound) and np.isfinite(upper_bound)
            else None
        ),
        "cuts_before": cuts_before,
        "cuts_after": len(cuts),
        "cut_rhs": float(rhs),
        "cut_tightness_error": float(cut_tightness_error),
        "master_time_seconds": float(master_time),
        "subproblem_time_seconds": float(subproblem_time),
        "iteration_time_seconds": float(iteration_time),
    })

    iteration += 1
    pbar_outer.update(1)

    # ------------------------------------------------------------------
    # Check convergence
    # ------------------------------------------------------------------
    if np.isfinite(lower_bound) and np.isfinite(upper_bound):
        gap = upper_bound - lower_bound
        tolerance = gap_tolerance * max(1.0, abs(upper_bound))
        print(
            f"Iteration {iteration}: "
            f"LB={lower_bound:.2f}  "
            f"UB={upper_bound:.2f}  "
            f"gap={gap:.2f}  "
            f"tol={tolerance:.2e}"
        )
        if gap <= tolerance:
            print("Converged.")
            break
    else:
        gap = np.inf
        print(
            f"Iteration {iteration}: "
            f"LB={lower_bound:.2f}  "
            f"UB={upper_bound:.2f}  "
            f"gap=inf"
        )

pbar_outer.close()

# ---------------------------------------------------------
# Extract and Log Results
# ---------------------------------------------------------

found_solution = best_u is not None

converged = (
    np.isfinite(lower_bound)
    and np.isfinite(upper_bound)
    and (upper_bound - lower_bound)
       <= gap_tolerance * max(1.0, abs(upper_bound))
)

cost_of_solution = (
    float(round(upper_bound, 2))
    if found_solution
    else None
)

final_gap = (
    float(upper_bound - lower_bound)
    if np.isfinite(lower_bound) and np.isfinite(upper_bound)
    else None
)

print("\n[Results Summary]")
print(f"  -> Solver Status:        {'Converged' if converged else 'Stopped'}")
print(f"  -> Found Solution?       {found_solution}")

if found_solution:
    print(f"  -> Best Expected Cost:   ${cost_of_solution:,.2f}")
else:
    print(f"  -> Best Expected Cost:   N/A")

print(
    f"  -> Final Lower Bound:    "
    f"${lower_bound:,.2f}" if np.isfinite(lower_bound)
    else "  -> Final Lower Bound:    N/A"
)

print(
    f"  -> Final Upper Bound:    "
    f"${upper_bound:,.2f}" if np.isfinite(upper_bound)
    else "  -> Final Upper Bound:    N/A"
)

print(
    f"  -> Final Gap:            "
    f"${final_gap:,.2f}" if final_gap is not None
    else "  -> Final Gap:            N/A"
)

print(f"  -> Iterations:            {iteration}")
print(f"  -> Cuts Generated:        {len(cuts)}")
print(f"  -> Algorithm Converged?   {converged}")

results_payload = {
    "metadata": {
        "experiment_name": "4-Generator Stochastic Unit Commitment",
        "solver_name": "Benders Decomposition with PuLP CBC",
        "seed_used": int(seed),
    },
    "core_required_metrics": {
        "found_solution": found_solution,
        "cost_of_solution": cost_of_solution,
        "converged": converged,
        "final_lower_bound": (
            float(lower_bound)
            if np.isfinite(lower_bound)
            else None
        ),
        "final_upper_bound": (
            float(upper_bound)
            if np.isfinite(upper_bound)
            else None
        ),
        "final_gap": final_gap,
        "iterations": int(iteration),
        "cuts_generated": int(len(cuts)),
    },
    "iteration_history": iteration_history,
}

os.makedirs("results", exist_ok=True)
json_path = os.path.join("results", "pulp_benders_suc.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(results_payload, f, indent=4)

print(f"\n[SUCCESS] Comprehensive JSON logged to: '{json_path}'")
print("==========================================================================\n")