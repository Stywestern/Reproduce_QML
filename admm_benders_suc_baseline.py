# ###########################################################################################################################################
# Imports
# ###########################################################################################################################################

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
from src.problems.generated_problems import g4t24s3_suc_problem, tiny_suc_problem

# ###########################################################################################################################################
# 1. Load the problem
# ###########################################################################################################################################

# We need to construct this model so that we can get appropriate variables, because then we will create another MILP model to seperate Master problem and Subproblem
seed = 2
#pd = g4t24s3_suc_problem(seed)
pd = tiny_suc_problem(seed)
model = SUCModelBuilder(pd).get_base_model()
first_stage_names = set(model.var_names[i] for i in model.first_stage_vars()) # This is a list of strings that hold u_i_t vars, 96 of them

print_timed("Done")

# ###########################################################################################################################################
# Helper Functions and Classes For The Loop
# ###########################################################################################################################################

@dataclass
class BendersCut:
    """
    Scenario-specific Benders optimality cut.

    Upsilon_s - theta_s^T u >= rhs_s
    """
    theta: dict[str, float]   # u-variable name -> aggregated dual
    rhs: float                # aggregate_subproblem_cost - sum(theta * u_l)
    iteration: int            # which BD iteration l produced this cut
    scenario: int             # which scenario this cut belongs to

dummy_cut = BendersCut({"u_gen3_hr1": 2718.28, "u_gen3_hr2": 3141.59}, 2023, -1, -1)

# ############################################################################################################################################
# Math Model Instantiations
# ############################################################################################################################################

# ============================================================================
# Pre-build Master
# ============================================================================

print_timed("[2. Instantiating Mathematical Model]")

upsilon_lower = 0

master_builder = MILPBuilder()
master_prob = pulp.LpProblem("Benders_Master", pulp.LpMinimize)
master_pulp_vars: dict[str, pulp.LpVariable] = {}

# Add variables to both the builder and PuLP
for i in model.first_stage_vars():
    name = model.var_names[i]
    master_builder.add_var(name, "B", 0.0, 1.0)
    
    master_pulp_vars[name] = pulp.LpVariable(name, lowBound=0.0, upBound=1.0, cat=pulp.LpBinary)
    
    if model.c[i] != 0:
        master_builder.add_obj(name, float(model.c[i]))

# Add Upsilon
upsilon_names = [f"Upsilon_s{s}" for s in range(pd.num_scenarios)]

for uname in upsilon_names:
    master_builder.add_var(uname, "C", upsilon_lower, np.inf)
    master_builder.add_obj(uname, 1.0)
    master_pulp_vars[uname] = pulp.LpVariable(uname, lowBound=upsilon_lower, cat=pulp.LpContinuous)

master_prob += pulp.lpSum(float(model.c[i]) * master_pulp_vars[model.var_names[i]] for i in model.first_stage_vars() if model.c[i] != 0
) + pulp.lpSum(master_pulp_vars[uname] for uname in upsilon_names)

# Add shared constraints to both
for r in model.shared_rows():
    row = model.A.getrow(r)
    terms_dict = {model.var_names[j]: float(v) for j, v in zip(row.indices, row.data)}
    sense, rhs = model.row_sense_rhs(r)
    
    master_builder.add_row(terms_dict, sense, rhs, model.constraint_labels[r])
    
    # Translate to PuLP expression
    expr = pulp.lpSum(coeff * master_pulp_vars[name] for name, coeff in terms_dict.items())
    label = model.constraint_labels[r]
    
    if sense == "=":
        master_prob += (expr == rhs, label)
    elif sense == "<=":
        master_prob += (expr <= rhs, label)
    else:
        master_prob += (expr >= rhs, label)

master_model = master_builder.build()

# ============================================================================
# Pre-build Base Subproblems
# ============================================================================
u_fixed_prefix = "ufix_"
base_sub_probs: list[pulp.LpProblem] = []
base_sub_vars: list[dict[str, pulp.LpVariable]] = []

for scenario in range(pd.num_scenarios):
    sub_prob = pulp.LpProblem(f"Benders_Base_Subproblem_s{scenario}", pulp.LpMinimize)
    sub_pulp_vars: dict[str, pulp.LpVariable] = {}
    
    # 1. Add continuous second-stage operational variables
    for i in model.scenario_vars(scenario):
        name = model.var_names[i]
        lb = None if model.lower[i] == -np.inf else float(model.lower[i])
        ub = None if model.upper[i] == np.inf else float(model.upper[i])
        sub_pulp_vars[name] = pulp.LpVariable(name, lowBound=lb, upBound=ub, cat=pulp.LpContinuous)
        
    # 2. Add continuous proxy variables for the first-stage commitments
    for i in model.first_stage_vars():
        name = model.var_names[i]
        proxy_name = u_fixed_prefix + name
        sub_pulp_vars[proxy_name] = pulp.LpVariable(proxy_name, lowBound=0.0, upBound=1.0, cat=pulp.LpContinuous)
        
    # 3. Add scenario objective
    sub_prob += pulp.lpSum(
        float(model.c[i]) * sub_pulp_vars[model.var_names[i]]
        for i in model.scenario_vars(scenario)
        if model.c[i] != 0
    )
    
    # 4. Add physical scenario constraints (load, limits, etc.)
    for r in model.scenario_rows(scenario):
        row = model.A.getrow(r)
        
        # Map variables: use proxy name if it's a stage-1 variable, otherwise use standard name
        expr = pulp.lpSum(
            float(v) * sub_pulp_vars[
                u_fixed_prefix + model.var_names[j] if model.var_names[j] in first_stage_names else model.var_names[j]
            ]
            for j, v in zip(row.indices, row.data)
        )
        
        sense, rhs = model.row_sense_rhs(r)
        label = model.constraint_labels[r]
        
        if sense == "=":
            sub_prob += (expr == rhs, label)
        elif sense == "<=":
            sub_prob += (expr <= rhs, label)
        else:
            sub_prob += (expr >= rhs, label)
            
    # Store the base problem and variables. We don't add fixing constraints yet, because PuLP can't handle them
    base_sub_probs.append(sub_prob)
    base_sub_vars.append(sub_pulp_vars)

print_timed("Done")

# ############################################################################################################################################
# Main While Loop
# ############################################################################################################################################

max_iterations = 100
gap_tolerance = 1e-4
lbs = []
ubs = []

cuts: list[BendersCut] = []
 
lower_bound = -np.inf
upper_bound = np.inf
best_u = None
iteration = 0

pbar_outer = tqdm(total=max_iterations, desc="Benders", unit="iter")

iteration_history = []
while iteration < max_iterations:
    iteration_start = time.perf_counter()

    # ------------------------------------------------------------------
    # SECTION I: Add cuts to the master
    # ------------------------------------------------------------------
    if cuts:
        latest_iter = cuts[-1].iteration
        latest_cuts = [c for c in cuts if c.iteration == latest_iter]
        for c in latest_cuts:
            print_timed(f"Integrating Cut (scen {c.scenario}, iter {c.iteration}, RHS: {c.rhs:.2f})")

            terms = {name: -c.theta[name] for name in c.theta}
            uname = f"Upsilon_s{c.scenario}"
            terms[uname] = 1.0
            master_model.add_row(terms, ">=", c.rhs, f"benders_cut_l{c.iteration}_s{c.scenario}")

            cut_expr = pulp.lpSum(-c.theta[name] * master_pulp_vars[name] for name in c.theta) + master_pulp_vars[uname]
            master_prob += (cut_expr >= c.rhs, f"benders_cut_l{c.iteration}_s{c.scenario}")

    # ------------------------------------------------------------------
    # SECTION II: Solve the master problem
    # ------------------------------------------------------------------
    if best_u is not None:
        for name, val in best_u.items():
            master_pulp_vars[name].setInitialValue(val)

    print_timed("Master solving...")
    master_start = time.perf_counter()
    master_solution = master_prob.solve(pulp.PULP_CBC_CMD(msg=False, gapRel=1e-2, warmStart=True, keepFiles=True)) # timeLimit=100, gapRel=0.05, 
    print_timed("Master solving complete")
    master_time = time.perf_counter() - master_start

    # The decisions of the master, which to open and which not to
    u_values = {name: int(round(var.varValue)) for name, var in master_pulp_vars.items() if name not in upsilon_names}
    upsilon_values = {uname: master_pulp_vars[uname].varValue for uname in upsilon_names}
    
    fixed_cost = sum(float(model.c[i]) * u_values[model.var_names[i]] for i in model.first_stage_vars() if model.c[i] != 0)
    lower_bound = float(pulp.value(master_prob.objective))
    lbs.append(lower_bound)

    # ------------------------------------------------------------------
    # SECTION III: Solve the subproblem for every scenario
    # ------------------------------------------------------------------

    expected_recourse_cost = 0.0
    new_cuts_this_iter = []

    subproblem_start = time.perf_counter()
    for scenario_num in tqdm(range(pd.num_scenarios), desc=f"Iteration {iteration} - Subproblems", unit="scenario", leave=False):
        # Clone the base problem structure instantly (preserves memory reference to variables)
        sub_prob = base_sub_probs[scenario_num].copy()
        sub_pulp_vars = base_sub_vars[scenario_num]
        
        # Inject the fixing constraints specific to this iteration's master solution
        for i in model.first_stage_vars():
            name = model.var_names[i]
            proxy_name = u_fixed_prefix + name
            sub_prob += (sub_pulp_vars[proxy_name] == float(u_values[name]), f"fix_{name}")

        sub_status_code = sub_prob.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=30))
        sub_status = pulp.LpStatus[sub_status_code]
        sub_cost = float(pulp.value(sub_prob.objective)) if sub_status == "Optimal" else None
 
        if sub_status != "Optimal":
            raise RuntimeError(f"Scenario {scenario} subproblem was not optimal: {sub_status}")
 
        # Calculate costs
        sub_cost = float(pulp.value(sub_prob.objective))
        expected_recourse_cost += sub_cost
        
        # Extract dual variables for cut generation
        theta_s = {}
        for i in model.first_stage_vars():
            name = model.var_names[i]
            theta_s[name] = sub_prob.constraints[f"fix_{name}"].pi

        rhs_s = sub_cost - sum(theta_s[name] * u_values[name] for name in theta_s)
        new_cuts_this_iter.append(BendersCut(theta=theta_s, rhs=rhs_s, iteration=iteration, scenario=scenario_num))

    cuts_before = len(cuts)
    cuts.extend(new_cuts_this_iter)
    subproblem_time = time.perf_counter() - subproblem_start

    # ------------------------------------------------------------------
    # SECTION IV: Update the Upper Bound
    # ------------------------------------------------------------------
    cuts_after = len(cuts)

    candidate_upper_bound = fixed_cost + expected_recourse_cost
    if candidate_upper_bound < upper_bound:
        upper_bound = candidate_upper_bound
        best_u = dict(u_values)

    ubs.append(upper_bound)

    if lower_bound > upper_bound + 1e-6:
        raise RuntimeError(f"Invalid Benders bounds: LB={lower_bound}, UB={upper_bound}")
    gap = upper_bound - lower_bound

    # Concise one-line iteration summary
    print_timed(
        f"Iter {iteration:3d} | LB: {lower_bound:12.2f} | UB: {upper_bound:12.2f} | "
        f"Gap: {gap:12.2f} | M-Time: {master_time:.2f}s | Sub-Time: {subproblem_time:.2f}s"
    )

    # ------------------------------------------------------------------
    # Log iterations
    # ------------------------------------------------------------------
    iteration_time = time.perf_counter() - iteration_start
    iteration_history.append({
        "iteration": int(iteration),
        "master_objective": float(lower_bound),
        "fixed_cost": float(fixed_cost),
        "recourse_cost": float(expected_recourse_cost),
        "candidate_upper_bound": float(candidate_upper_bound),
        "global_lower_bound": float(lower_bound) if np.isfinite(lower_bound) else None,
        "global_upper_bound": float(upper_bound) if np.isfinite(upper_bound) else None,
        "absolute_gap": float(gap) if np.isfinite(gap) else None,
        "cuts_before": cuts_before,
        "cuts_after": len(cuts),
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
print("lower bounds:", lbs)
print("upper bounds:", ubs)
print("cuts", cuts)

found_solution = best_u is not None

solution_details = {}
if found_solution:
    sorted_keys = sorted(best_u.keys())
    bitstring = "".join(str(best_u[k]) for k in sorted_keys)
    
    generator_schedules = {}
    for name, val in best_u.items():
        if val == 1:
            parts = name.split("_")
            gen_id = parts[1]
            hour_id = int(parts[2].replace("hr", ""))
            if gen_id not in generator_schedules:
                generator_schedules[gen_id] = []
            generator_schedules[gen_id].append(hour_id)
            
    for gen_id in generator_schedules:
        generator_schedules[gen_id].sort()

    solution_details = {
        "commitment_bitstring": bitstring,
        "raw_u_values": best_u,
        "active_hours_per_generator": generator_schedules
    }

converged = (
    np.isfinite(lower_bound)
    and np.isfinite(upper_bound)
    and (upper_bound - lower_bound)
       <= gap_tolerance * max(1.0, abs(upper_bound))
)

final_expected_cost = None
if best_u is not None:
    final_expected_recourse = 0.0
    final_scenario_costs = {}

    for scenario_num in range(pd.num_scenarios):
        sub_prob = base_sub_probs[scenario_num].copy()
        sub_pulp_vars = base_sub_vars[scenario_num]
        
        for i in model.first_stage_vars():
            name = model.var_names[i]
            proxy_name = u_fixed_prefix + name
            sub_prob += (sub_pulp_vars[proxy_name] == float(best_u[name]), f"fix_{name}")

        sub_prob.solve(pulp.PULP_CBC_CMD(msg=False))
        weighted_scen_cost = float(pulp.value(sub_prob.objective)) # Already includes prob
        
        prob = pd.scenario_probs[scenario_num]
        raw_scen_cost = weighted_scen_cost / prob if prob > 0 else 0.0
        
        final_expected_recourse += weighted_scen_cost # DO NOT multiply by prob again!
        final_scenario_costs[f"scenario_{scenario_num}"] = round(raw_scen_cost, 2)

    final_fixed_cost = sum(
        float(model.c[i]) * best_u[model.var_names[i]] 
        for i in model.first_stage_vars() if model.c[i] != 0
    )
    
    final_expected_cost = final_fixed_cost + final_expected_recourse

final_gap = (
    float(upper_bound - lower_bound)
    if np.isfinite(lower_bound) and np.isfinite(upper_bound)
    else None
)

print("\n[Results Summary]")
print(f"  -> Solver Status:         {'Converged' if converged else 'Stopped'}")
print(f"  -> Found Solution?        {found_solution}")

if found_solution:
    print(f"  -> Best Expected Cost:    ${final_expected_cost:,.2f}")
else:
    print(f"  -> Best Expected Cost:    N/A")

print(
    f"  -> Final Lower Bound:     "
    f"${lower_bound:,.2f}" if np.isfinite(lower_bound)
    else "  -> Final Lower Bound:     N/A"
)

print(
    f"  -> Final Upper Bound:     "
    f"${upper_bound:,.2f}" if np.isfinite(upper_bound)
    else "  -> Final Upper Bound:     N/A"
)

print(
    f"  -> Final Gap:             "
    f"${final_gap:,.2f}" if final_gap is not None
    else "  -> Final Gap:             N/A"
)

print(f"  -> Iterations:            {iteration}")
print(f"  -> Cuts Generated:        {len(cuts)}")
print(f"  -> Algorithm Converged?   {converged}")