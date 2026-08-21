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
import math

import dimod
import neal
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

seed = 2
pd, prob_name = g4t24s3_suc_problem(seed)

model = SUCModelBuilder(pd).get_base_model()
first_stage_names = set(model.var_names[i] for i in model.first_stage_vars())

# ###########################################################################################################################################
# Helper Functions and Classes For The Loop
# ###########################################################################################################################################

@dataclass
class BendersCut:
    theta: dict[str, float]   
    rhs: float                
    iteration: int                       

# ############################################################################################################################################
# Math Model Instantiations
# ############################################################################################################################################

print_timed("[1. Instantiating Math Models]")

max_load = pd.demand.max()
max_upsilon = int(max_load * pd.num_hours * pd.shed_cost)
J_bits = int(math.ceil(math.log2(max_upsilon))) if max_upsilon > 0 else 1

cqm = dimod.ConstrainedQuadraticModel()
u_vars = {model.var_names[i]: dimod.Binary(model.var_names[i]) for i in model.first_stage_vars()}

# Explicit binary encoding for a single Upsilon
upsilon_vars = {}
all_binary_vars = list(u_vars.keys())

# --- SINGLE CUT: One Global Upsilon ---
bits = {f"ups_total_b{j}": dimod.Binary(f"ups_total_b{j}") for j in range(J_bits)}
all_binary_vars.extend(bits.keys())
upsilon_vars["Upsilon_Total"] = sum((2**j) * bits[f"ups_total_b{j}"] for j in range(J_bits))

GLOBAL_SCALE = 10000.0

# Objective (Scaled)
obj = sum((float(model.c[i]) / GLOBAL_SCALE) * u_vars[model.var_names[i]] for i in model.first_stage_vars() if model.c[i] != 0)
obj += sum((1.0 / GLOBAL_SCALE) * upsilon_vars[uname] for uname in upsilon_vars)
cqm.set_objective(obj)

# Add shared constraints directly to CQM 
for r in model.shared_rows():
    row = model.A.getrow(r)
    expr = sum(float(v) * u_vars[model.var_names[j]] for j, v in zip(row.indices, row.data))
    sense, rhs = model.row_sense_rhs(r)
    
    if sense == "=": cqm.add_constraint(expr == rhs, label=model.constraint_labels[r])
    elif sense == "<=": cqm.add_constraint(expr <= rhs, label=model.constraint_labels[r])
    else: cqm.add_constraint(expr >= rhs, label=model.constraint_labels[r])

# ============================================================================
# Pre-build Base Subproblems (Classical PuLP)
# ============================================================================
u_fixed_prefix = "ufix_"
base_sub_probs: list[pulp.LpProblem] = []
base_sub_vars: list[dict[str, pulp.LpVariable]] = []

for scenario in range(pd.num_scenarios):
    sub_prob = pulp.LpProblem(f"Benders_Base_Subproblem_s{scenario}", pulp.LpMinimize)
    sub_pulp_vars: dict[str, pulp.LpVariable] = {}
    
    for i in model.scenario_vars(scenario):
        name = model.var_names[i]
        lb = None if model.lower[i] == -np.inf else float(model.lower[i])
        ub = None if model.upper[i] == np.inf else float(model.upper[i])
        sub_pulp_vars[name] = pulp.LpVariable(name, lowBound=lb, upBound=ub, cat=pulp.LpContinuous)
        
    for i in model.first_stage_vars():
        name = model.var_names[i]
        proxy_name = u_fixed_prefix + name
        sub_pulp_vars[proxy_name] = pulp.LpVariable(proxy_name, lowBound=0.0, upBound=1.0, cat=pulp.LpContinuous)
        
    sub_prob += pulp.lpSum(
        float(model.c[i]) * sub_pulp_vars[model.var_names[i]]
        for i in model.scenario_vars(scenario) if model.c[i] != 0
    )
    
    for r in model.scenario_rows(scenario):
        row = model.A.getrow(r)
        expr = pulp.lpSum(
            float(v) * sub_pulp_vars[u_fixed_prefix + model.var_names[j] if model.var_names[j] in first_stage_names else model.var_names[j]]
            for j, v in zip(row.indices, row.data)
        )
        sense, rhs = model.row_sense_rhs(r)
        label = model.constraint_labels[r]
        
        if sense == "=": sub_prob += (expr == rhs, label)
        elif sense == "<=": sub_prob += (expr <= rhs, label)
        else: sub_prob += (expr >= rhs, label)
            
    base_sub_probs.append(sub_prob)
    base_sub_vars.append(sub_pulp_vars)

print_timed("Done")

# ############################################################################################################################################
# Main While Loop
# ############################################################################################################################################

max_iterations = 100
gap_tolerance = 1e-2
lbs = []
ubs = []
iteration_history = []

cuts: list[BendersCut] = []
qubit_history = []
 
lower_bound = -np.inf
upper_bound = np.inf
best_u = None
iteration = 0

u_core = {name: 0.5 for name in first_stage_names}
sampler = neal.SimulatedAnnealingSampler()

print_timed("[2. Main Loop]")
pbar_outer = tqdm(total=max_iterations, desc="Benders", unit="iter", disable=False)

while iteration < max_iterations:
    iteration_start = time.perf_counter()

    # ------------------------------------------------------------------
    # SECTION I: Add cuts to the master via CQM
    # ------------------------------------------------------------------
    if cuts:
        latest_iter = cuts[-1].iteration
        latest_cuts = [c for c in cuts if c.iteration == latest_iter]
        for c in latest_cuts:
            uname = "Upsilon_Total"
            
            cut_expr = sum((float(-c.theta[name]) / GLOBAL_SCALE) * u_vars[name] for name in c.theta) 
            cut_expr += (1.0 / GLOBAL_SCALE) * upsilon_vars[uname]
            rhs_scaled = float(c.rhs) / GLOBAL_SCALE
            
            cqm.add_constraint(cut_expr >= rhs_scaled, label=f"benders_cut_l{c.iteration}_global")

    # ------------------------------------------------------------------
    # SECTION II: ADMM Master Problem Solve (Native Penalties / NO QPHR)
    # ------------------------------------------------------------------
    master_start = time.perf_counter()
    
    # Convert CQM to BQM natively - this automatically generates slack variables for inequalities
    bqm, invert_mapping = dimod.cqm_to_bqm(cqm)
    
    # Maintain state across iterations, appending any newly generated slack variables
    if 'best_sample' not in locals():
        best_sample = {v: 0 for v in bqm.variables}
    else:
        for v in bqm.variables:
            if v not in best_sample:
                best_sample[v] = 0

    # Define the core ADMM Blocks
    admm_blocks = []
    for g in range(pd.num_gens):
        admm_blocks.append([f"u_gen{g}_hr{t}" for t in range(pd.num_hours)])
        
    ups_vars_block = [f"ups_total_b{j}" for j in range(J_bits)]
    admm_blocks.append(ups_vars_block)

    # Collect all native dimod slack variables into a final block
    known_vars = set([v for block in admm_blocks for v in block])
    slack_block = [v for v in bqm.variables if v not in known_vars]
    if slack_block:
        admm_blocks.append(slack_block)

    max_qubits_used = 0
    max_inner_iters = 10
    
    for inner_iter in range(max_inner_iters):
        for block_idx, block in enumerate(admm_blocks):
            if not block:
                continue
                
            # Build partial BQM by freezing off-block variables
            sub_bqm = bqm.copy()
            fixed_vars = {v: best_sample[v] for v in sub_bqm.variables if v not in block}
            sub_bqm.fix_variables(fixed_vars)
            
            current_qubits = len(sub_bqm.variables)
            if current_qubits > max_qubits_used:
                max_qubits_used = current_qubits
            
            # Solve only the small localized BQM
            if current_qubits > 0:
                sampleset = sampler.sample(sub_bqm, num_reads=250, num_sweeps=1000)
                best_sub_sample = sampleset.first.sample
                
                # Update global state
                for v in block:
                    best_sample[v] = int(best_sub_sample.get(v, 0))
                
            if inner_iter == max_inner_iters - 1:
                block_type = f"Gen {block_idx}" if block_idx < pd.num_gens else ("Upsilon" if block_idx == pd.num_gens else "Slack Variables")

    qubit_history.append(max_qubits_used)
    master_time = time.perf_counter() - master_start

    # Extract base costs natively
    u_values = {name: int(round(best_sample[name])) for name in u_vars}
    fixed_cost = sum(float(model.c[i]) * u_values[model.var_names[i]] for i in model.first_stage_vars() if model.c[i] != 0)
    ups_sum = sum((2**j) * best_sample[f"ups_total_b{j}"] for j in range(J_bits))
        
    base_obj_val = fixed_cost + ups_sum
    
    if base_obj_val > upper_bound and np.isfinite(upper_bound):
        lower_bound = upper_bound
    else:
        lower_bound = base_obj_val
        
    lbs.append(lower_bound)

    # ------------------------------------------------------------------
    # SECTION III: Solve the subproblem for every scenario
    # ------------------------------------------------------------------

    expected_recourse_cost = 0.0
    new_cuts_this_iter = []

    if best_u is not None:
        u_core = {name: 0.5 * u_core[name] + 0.5 * u_values[name] for name in first_stage_names}

    subproblem_start = time.perf_counter()
    
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

        sub_prob_cut.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=30))
        core_sub_cost = float(pulp.value(sub_prob_cut.objective))

        theta_s = {model.var_names[i]: sub_prob_cut.constraints[f"fix_{model.var_names[i]}"].pi for i in model.first_stage_vars()}

        rhs_s = core_sub_cost - sum(theta_s[name] * u_core[name] for name in theta_s)
        
        total_rhs += rhs_s
        for name in theta_s:
            total_theta[name] += theta_s[name]

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

    iteration_time = time.perf_counter() - iteration_start
    iteration_history.append({
        "iteration": int(iteration),
        "qubits_needed": int(current_qubits),
        "master_objective": float(lower_bound),
        "fixed_cost": float(fixed_cost),
        "recourse_cost": float(expected_recourse_cost),
        "candidate_upper_bound": float(candidate_upper_bound),
        "global_lower_bound": float(lower_bound),
        "global_upper_bound": float(upper_bound),
        "master_time_seconds": float(master_time),
        "subproblem_time_seconds": float(subproblem_time),
        "iteration_time_seconds": float(iteration_time),
    })

    iteration += 1
    pbar_outer.update(1)

    if np.isfinite(lower_bound) and np.isfinite(upper_bound):
        gap = upper_bound - lower_bound
        tolerance = gap_tolerance * max(1.0, abs(upper_bound))
        
        print_timed(
            f"Iter {iteration:3d} | LB: {lower_bound:12.2f} | UB: {upper_bound:12.2f} | "
            f"Gap: {gap:12.2f} | Max Block Qubits: {max_qubits_used}"
        )
        
        if gap <= tolerance:
            print_timed("Converged.")
            break
    else:
        gap = np.inf
        print_timed(f"Iter {iteration:3d} | LB: {lower_bound:.2f} | UB: {upper_bound:.2f} | Gap: inf | Max Block Qubits: {max_qubits_used}")

    if max_qubits_used > 1000:
        print_timed(f"Hardware Limit Reached: Qubit count ({max_qubits_used}) exceeds 1000. Terminating proof of concept.")
        break

pbar_outer.close()
