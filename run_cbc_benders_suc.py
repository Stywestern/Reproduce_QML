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
from src.problems.generated_problems import g4t24s3_suc_problem, g10t24s10_suc_problem

# ###########################################################################################################################################
# 1. Load the problem
# ###########################################################################################################################################

# We need to construct this model so that we can get appropriate variables, because then we will create another MILP model to seperate Master problem and Subproblem
seed = 2
pd, prob_name = g4t24s3_suc_problem(seed)
#pd, prob_name = g10t24s10_suc_problem(seed)

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

dummy_cut = BendersCut({"u_gen3_hr1": 2718.28, "u_gen3_hr2": 3141.59}, 2023, -1)

# ############################################################################################################################################
# Math Model Instantiations
# ############################################################################################################################################

# ============================================================================
# Pre-build Master
# ============================================================================

print_timed("[1. Instantiating Mathematical Model]")

upsilon_lower = 0

master_builder = MILPBuilder()
master_prob = pulp.LpProblem("Benders_Master", pulp.LpMinimize)
master_pulp_vars: dict[str, pulp.LpVariable] = {}

for i in model.first_stage_vars():
    name = model.var_names[i]
    master_builder.add_var(name, "B", 0.0, 1.0)
    master_pulp_vars[name] = pulp.LpVariable(name, lowBound=0.0, upBound=1.0, cat=pulp.LpBinary)
    if model.c[i] != 0:
        master_builder.add_obj(name, float(model.c[i]))

upsilon_names = ["Upsilon_Total"]

for uname in upsilon_names:
    master_builder.add_var(uname, "C", upsilon_lower, np.inf)
    master_builder.add_obj(uname, 1.0)
    master_pulp_vars[uname] = pulp.LpVariable(uname, lowBound=upsilon_lower, cat=pulp.LpContinuous)

master_prob += pulp.lpSum(
    float(model.c[i]) * master_pulp_vars[model.var_names[i]] for i in model.first_stage_vars() if model.c[i] != 0
) + pulp.lpSum(master_pulp_vars[uname] for uname in upsilon_names)

for r in model.shared_rows():
    row = model.A.getrow(r)
    terms_dict = {model.var_names[j]: float(v) for j, v in zip(row.indices, row.data)}
    sense, rhs = model.row_sense_rhs(r)
    
    master_builder.add_row(terms_dict, sense, rhs, model.constraint_labels[r])
    
    expr = pulp.lpSum(coeff * master_pulp_vars[name] for name, coeff in terms_dict.items())
    label = model.constraint_labels[r]
    
    if sense == "=": master_prob += (expr == rhs, label)
    elif sense == "<=": master_prob += (expr <= rhs, label)
    else: master_prob += (expr >= rhs, label)

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
gap_tolerance = 1e-2 # for convergence
lbs = []
ubs = []

cuts: list[BendersCut] = []
 
lower_bound = -np.inf
upper_bound = np.inf
best_u = None
iteration = 0

u_core = {name: 0.5 for name in first_stage_names}
pbar_outer = tqdm(total=max_iterations, desc="Benders", unit="iter")

print_timed("[2. Main Loop]")
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
            uname = "Upsilon_Total"
            terms = {name: -c.theta[name] for name in c.theta}
            terms[uname] = 1.0
            
            master_model.add_row(terms, ">=", c.rhs, f"benders_cut_l{c.iteration}_global")
            cut_expr = pulp.lpSum(-c.theta[name] * master_pulp_vars[name] for name in c.theta) + master_pulp_vars[uname]
            master_prob += (cut_expr >= c.rhs, f"benders_cut_l{c.iteration}_global")

    # ------------------------------------------------------------------
    # SECTION II: Solve the master problem
    # ------------------------------------------------------------------
    if best_u is not None:
        for name, val in best_u.items():
            master_pulp_vars[name].setInitialValue(val)

    current_benders_gap = (upper_bound - lower_bound) / max(1.0, abs(upper_bound)) if np.isfinite(upper_bound) else 1.0
    dynamic_master_gap = max(0.01, min(0.10, current_benders_gap * 0.5))

    master_start = time.perf_counter()
    master_solution = master_prob.solve(pulp.PULP_CBC_CMD(msg=False, gapRel=dynamic_master_gap, warmStart=True, keepFiles=True)) 
    master_time = time.perf_counter() - master_start

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

    if best_u is not None:
        u_core = {name: 0.5 * u_core[name] + 0.5 * u_values[name] for name in first_stage_names}

    subproblem_start = time.perf_counter()
    
    # --- Single-Cut Aggregation Variables ---
    total_theta = {name: 0.0 for name in first_stage_names}
    total_rhs = 0.0

    for scenario_num in range(pd.num_scenarios):
        # --- PHASE A: Valuation ---
        sub_prob_val = base_sub_probs[scenario_num].copy()
        sub_pulp_vars_val = base_sub_vars[scenario_num]
        
        for i in model.first_stage_vars():
            name = model.var_names[i]
            proxy_name = u_fixed_prefix + name
            sub_prob_val += (sub_pulp_vars_val[proxy_name] == float(u_values[name]), f"fix_{name}")

        sub_status_code_val = sub_prob_val.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=30))
        if pulp.LpStatus[sub_status_code_val] != "Optimal":
            raise RuntimeError(f"Scenario {scenario_num} valuation subproblem failed.")
            
        actual_sub_cost = float(pulp.value(sub_prob_val.objective))
        expected_recourse_cost += actual_sub_cost

        # --- PHASE B: Core Point Duals ---
        sub_prob_cut = base_sub_probs[scenario_num].copy()
        sub_pulp_vars_cut = base_sub_vars[scenario_num]
        
        for i in model.first_stage_vars():
            name = model.var_names[i]
            proxy_name = u_fixed_prefix + name
            sub_prob_cut += (sub_pulp_vars_cut[proxy_name] == float(u_core[name]), f"fix_{name}")

        sub_status_code_cut = sub_prob_cut.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=30))
        if pulp.LpStatus[sub_status_code_cut] != "Optimal":
            raise RuntimeError(f"Scenario {scenario_num} core-point subproblem failed.")
            
        core_sub_cost = float(pulp.value(sub_prob_cut.objective))

        theta_s = {}
        for i in model.first_stage_vars():
            name = model.var_names[i]
            theta_s[name] = sub_prob_cut.constraints[f"fix_{name}"].pi

        rhs_s = core_sub_cost - sum(theta_s[name] * u_core[name] for name in theta_s)
        
        # Aggregate the cut components
        total_rhs += rhs_s
        for name in theta_s:
            total_theta[name] += theta_s[name]

    # Create the single aggregated cut
    this_iter_cut = BendersCut(theta=total_theta, rhs=total_rhs, iteration=iteration)
    new_cuts_this_iter.append(this_iter_cut)

    cuts_before = len(cuts)
    cuts.extend(new_cuts_this_iter)
    subproblem_time = time.perf_counter() - subproblem_start

    # ------------------------------------------------------------------
    # SECTION IV: Update the Upper Bound
    # ------------------------------------------------------------------
    candidate_upper_bound = fixed_cost + expected_recourse_cost
    if candidate_upper_bound < upper_bound:
        upper_bound = candidate_upper_bound
        best_u = dict(u_values)

    ubs.append(upper_bound)

    if lower_bound > upper_bound + 1e-6:
        raise RuntimeError(f"Invalid Benders bounds: LB={lower_bound}, UB={upper_bound}")

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
        "cuts_before": cuts_before,
        "cuts_after": len(cuts),
        "master_time_seconds": float(master_time),
        "subproblem_time_seconds": float(subproblem_time),
        "iteration_time_seconds": float(iteration_time),
    })

    iteration += 1
    pbar_outer.update(1)

    if np.isfinite(lower_bound) and np.isfinite(upper_bound):
        gap = upper_bound - lower_bound
        tolerance = gap_tolerance * max(1.0, abs(upper_bound))
        print(f"Iteration {iteration}: LB={lower_bound:.2f}  UB={upper_bound:.2f}  gap={gap:.2f}  tol={tolerance:.2e}")
        if gap <= tolerance:
            print("Converged.")
            break
    else:
        gap = np.inf
        print(f"Iteration {iteration}: LB={lower_bound:.2f}  UB={upper_bound:.2f}  gap=inf")

pbar_outer.close()

# ---------------------------------------------------------
# Extract, Format, and Log Results
# ---------------------------------------------------------
import datetime

found_solution = best_u is not None
converged = (
    np.isfinite(lower_bound)
    and np.isfinite(upper_bound)
    and (upper_bound - lower_bound) <= gap_tolerance * max(1.0, abs(upper_bound))
)
status_str = "Optimal" if converged else ("Feasible" if found_solution else "Infeasible")
exec_time = globals().get('exec_time', sum(i['iteration_time_seconds'] for i in iteration_history) if 'iteration_history' in globals() else 0.0)
final_gap = float(upper_bound - lower_bound) if (np.isfinite(lower_bound) and np.isfinite(upper_bound)) else None

print(f"\n[3. Results Summary]")
print(f"  -> Solver Status:        {status_str}")
print(f"  -> Found Solution?       {found_solution}")
print(f"  -> Optimal Cost:         {f'${upper_bound:,.2f}' if found_solution else 'N/A'}")
print(f"  -> Final Lower Bound:    {f'${lower_bound:,.2f}' if np.isfinite(lower_bound) else 'N/A'}")
print(f"  -> Final Gap:            {f'${final_gap:,.2f}' if final_gap is not None else 'N/A'}")
print(f"  -> Time Taken:           {exec_time:.2f} seconds")
print(f"  -> Iterations / Cuts:    {iteration} / {len(cuts)}")
print(f"  -> Algorithm Converged?  {converged}")

schedule_by_generator = {f"gen{g}": [] for g in range(pd.num_gens)}
best_bitstring = ""
cost_by_scenario = {}
fixed_cost = dispatch_cost = shed_cost = spill_cost = 0.0
expected_recourse_cost = 0.0

if found_solution:
    u_matrix = np.zeros((pd.num_gens, pd.num_hours), dtype=int)
    for t in range(pd.num_hours):
        for g in range(pd.num_gens):
            val = int(best_u.get(f"u_gen{g}_hr{t}", 0))
            u_matrix[g, t] = val
            schedule_by_generator[f"gen{g}"].append(val)
    best_bitstring = "".join(u_matrix.flatten().astype(str))

    for i in model.first_stage_vars():
        name = model.var_names[i]
        fixed_cost += float(model.c[i]) * best_u[name]

    print(f"\n[6. Cost by Scenario]")
    for s in range(pd.num_scenarios):
        sub_prob = base_sub_probs[s].copy()
        sub_pulp_vars = base_sub_vars[s]
        
        for i in model.first_stage_vars():
            name = model.var_names[i]
            proxy_name = u_fixed_prefix + name
            sub_prob += (sub_pulp_vars[proxy_name] == float(best_u[name]), f"fix_{name}")

        sub_prob.solve(pulp.PULP_CBC_CMD(msg=False))
        
        weighted_scen_cost = 0.0
        for i in model.scenario_vars(s):
            name = model.var_names[i]
            val = sub_pulp_vars[name].varValue or 0.0
            term_cost = float(model.c[i]) * val
            
            weighted_scen_cost += term_cost
            if name.startswith("P_gen"): dispatch_cost += term_cost
            elif name.startswith("shed_"): shed_cost += term_cost
            elif name.startswith("spill_"): spill_cost += term_cost
            
        prob = pd.scenario_probs[s]
        raw_cost = weighted_scen_cost / prob if prob > 0 else 0.0
        cost_by_scenario[f"scenario_{s}"] = round(raw_cost, 2)
        expected_recourse_cost += weighted_scen_cost

        print(f"  -> Scenario {s}: P={prob:.3f}, Raw=${raw_cost:,.2f}, Weighted=${weighted_scen_cost:,.2f}")

    print(f"\n[4. Cost Breakdown]")
    print(f"  -> Fixed Costs:        ${fixed_cost:,.2f}")
    print(f"  -> Dispatch Costs:     ${dispatch_cost:,.2f}")
    print(f"  -> Shedding Penalty:   ${shed_cost:,.2f}")
    if spill_cost > 0:
        print(f"  -> Spilling Penalty:   ${spill_cost:,.2f}")
    print(f"  -> Total Computed:     ${(fixed_cost + expected_recourse_cost):,.2f}")
    print("-" * 50)
    print(f"  -> Expected Total Cost: ${(fixed_cost + expected_recourse_cost):,.2f}")

total_bits = pd.num_gens * pd.num_hours
total_vars = len(model.var_names)

results_payload = {
    "metadata": {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "experiment_name": "4-Generator Stochastic Unit Commitment - Single-Cut Benders",
        "solver_name": "PuLP CBC (Benders Master)",
        "seed_used": int(seed)
    },
    "core_required_metrics": {
        "found_solution": found_solution,
        "cost_of_solution": float(upper_bound) if found_solution else None,
        "time_it_took_seconds": float(exec_time),
        "converged": converged
    },
    "extended_solver_metrics": {
        "solver_status": status_str,
        "num_hours": int(pd.num_hours),
        "num_scenarios": int(pd.num_scenarios),
        "total_variables_count": int(total_vars),
        "binary_variables_count": int(total_bits),
        "continuous_variables_count": int(total_vars - total_bits),
        "total_constraints_count": int(len(model.constraint_labels)),
        "total_search_space_formula": f"2^{total_bits}",
        "benders_final_lower_bound": float(lower_bound) if np.isfinite(lower_bound) else None,
        "benders_final_upper_bound": float(upper_bound) if np.isfinite(upper_bound) else None,
        "benders_final_gap": float(final_gap) if final_gap is not None else None,
        "benders_iterations": int(iteration),
        "benders_cuts_generated": int(len(cuts))
    },
    "solution_details": {
        "best_bitstring": best_bitstring,
        "commitment_schedule_by_generator": schedule_by_generator,
        "cost_by_scenario": cost_by_scenario
    }
}

os.makedirs("results", exist_ok=True)
json_path = os.path.join("results", f"pulp_cbc_suc_benders_{prob_name}.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(results_payload, f, indent=4)

print(f"\n[SUCCESS] Comprehensive JSON logged to: '{json_path}'")