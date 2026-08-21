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
import time

import dimod

# --- QISKIT IMPORTS ---
from qiskit.primitives import StatevectorSampler
from qiskit_algorithms.optimizers import COBYLA
from qiskit_algorithms import QAOA
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer

def print_timed(*args, **kwargs):
    """Prints arguments prefixed with the current timestamp [HH:MM:SS]."""
    timestamp = time.strftime("[%H:%M:%S]")
    print(timestamp, *args, **kwargs)

from src.problems.suc_problem_generator import SUCProblemData, SUCModelBuilder, MILPModel, MILPBuilder
from src.problems.generated_problems import g4t24s3_suc_problem, g10t24s10_suc_problem

# ###########################################################################################################################################
# Helper Functions
# ###########################################################################################################################################

@dataclass
class BendersCut:
    theta: dict[str, float]   
    rhs: float                
    iteration: int            
    scenario: int             

def bqm_to_qiskit_qp(bqm: dimod.BinaryQuadraticModel) -> QuadraticProgram:
    """Translates a dimod BQM to a Qiskit QuadraticProgram."""
    qp = QuadraticProgram()
    
    for v in bqm.variables:
        qp.binary_var(name=str(v))
        
    lin = {str(k): float(v) for k, v in bqm.linear.items()}
    quad = {(str(u), str(v)): float(c) for (u, v), c in bqm.quadratic.items()}
    
    qp.minimize(linear=lin, quadratic=quad, constant=float(bqm.offset))
    return qp

# ###########################################################################################################################################
# QPHR-ALM Math Functions
# ###########################################################################################################################################

def parse_cqm_constraints(cqm: dimod.ConstrainedQuadraticModel) -> list[dict]:
    constraints_data = []
    for c_name, constraint in cqm.constraints.items():
        sense = constraint.sense.value
        if sense == '==': 
            continue
            
        linear_coeffs = {var: float(coeff) for var, coeff in constraint.lhs.linear.items()}
        constant_b = float(constraint.lhs.offset) - float(constraint.rhs)

        if sense == '>=':
            linear_coeffs = {v: -c for v, c in linear_coeffs.items()}
            constant_b = -constant_b

        constraints_data.append({'name': c_name, 'a': linear_coeffs, 'b': constant_b})
    return constraints_data

def evaluate_g_i(constraint: dict, sample: dict) -> float:
    return constraint['b'] + sum(coeff * int(sample.get(var, 0)) for var, coeff in constraint['a'].items())

def compute_residual_R(parsed_constraints: list, sample: dict, lambda_vec: np.ndarray, sigma: float) -> float:
    sum_sq = sum(max(-lambda_vec[i] / sigma, evaluate_g_i(c, sample))**2 for i, c in enumerate(parsed_constraints))
    return math.sqrt(sum_sq)

def build_qphr_admm_bqm(cqm: dimod.ConstrainedQuadraticModel, parsed_constraints: list, 
                        lambda_vec: np.ndarray, sigma: float, 
                        block_vars: list, current_sample: dict) -> dimod.BinaryQuadraticModel:
    lin = {}
    quad = {}
    offset = 0.0
    
    for v in block_vars:
        if v in cqm.objective.linear:
            lin[v] = float(cqm.objective.linear[v])
            
    for (u, v), coeff in cqm.objective.quadratic.items():
        if u in block_vars and v in block_vars:
            key = (u, v) if u < v else (v, u)
            quad[key] = quad.get(key, 0.0) + float(coeff)
        elif u in block_vars and v not in block_vars:
            lin[u] = lin.get(u, 0.0) + float(coeff) * current_sample[v]
        elif v in block_vars and u not in block_vars:
            lin[v] = lin.get(v, 0.0) + float(coeff) * current_sample[u]

    for i, constr in enumerate(parsed_constraints):
        lambda_i = lambda_vec[i]
        
        g_val = evaluate_g_i(constr, current_sample)
        if (lambda_i + sigma * g_val) > 0:
            b_eff = constr['b'] + sum(c * current_sample[v] for v, c in constr['a'].items() if v not in block_vars)
            K = sigma * b_eff + lambda_i
            
            block_a = {v: c for v, c in constr['a'].items() if v in block_vars}
            vars_list = list(block_a.keys())
            
            for j, v_j in enumerate(vars_list):
                C_j = sigma * block_a[v_j]
                lin[v_j] = lin.get(v_j, 0.0) + (C_j**2 + 2.0 * K * C_j) / (2.0 * sigma)
                
                for k in range(j + 1, len(vars_list)):
                    v_k = vars_list[k]
                    quad_term = sigma * block_a[v_j] * block_a[v_k]
                    key = (v_j, v_k) if v_j < v_k else (v_k, v_j)
                    quad[key] = quad.get(key, 0.0) + quad_term
                    
    return dimod.BinaryQuadraticModel(lin, quad, offset, 'BINARY')

# ############################################################################################################################################
# 1. Load the problem
# ############################################################################################################################################

seed = 2
pd, prob_name = g4t24s3_suc_problem(seed)

model = SUCModelBuilder(pd).get_base_model()
first_stage_names = set(model.var_names[i] for i in model.first_stage_vars())

# ############################################################################################################################################
# Math Model Instantiations
# ############################################################################################################################################

print_timed("[1. Instantiating Math Models]")

max_load = pd.demand.max()
max_upsilon = int(max_load * pd.num_hours * pd.shed_cost)
J_bits = int(math.ceil(math.log2(max_upsilon))) if max_upsilon > 0 else 1

cqm = dimod.ConstrainedQuadraticModel()
u_vars = {model.var_names[i]: dimod.Binary(model.var_names[i]) for i in model.first_stage_vars()}

upsilon_vars = {}
all_binary_vars = list(u_vars.keys())

# --- SINGLE CUT: One Global Upsilon ---
bits = {f"ups_total_b{j}": dimod.Binary(f"ups_total_b{j}") for j in range(J_bits)}
all_binary_vars.extend(bits.keys())
upsilon_vars["Upsilon_Total"] = sum((2**j) * bits[f"ups_total_b{j}"] for j in range(J_bits))

GLOBAL_SCALE = 10000.0

obj = sum((float(model.c[i]) / GLOBAL_SCALE) * u_vars[model.var_names[i]] for i in model.first_stage_vars() if model.c[i] != 0)
obj += sum((1.0 / GLOBAL_SCALE) * upsilon_vars[uname] for uname in upsilon_vars)
cqm.set_objective(obj)

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

# --- QAOA Setup ---

# Use the modern V2 StatevectorSampler
qiskit_sampler = StatevectorSampler()

# Limit maxiter severely to prevent days of execution time
optimizer = COBYLA(maxiter=30)
qaoa_mes = QAOA(sampler=qiskit_sampler, optimizer=optimizer, reps=1)
qaoa_optimizer = MinimumEigenOptimizer(qaoa_mes)

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
    # SECTION II: QPHR-ADMM Master Problem Solve
    # ------------------------------------------------------------------
    master_start = time.perf_counter()
    
    sigma, eta, rho, max_inner_iters = 0.01, 1.05, 0.9, 10
    
    if 'best_sample' not in locals():
        best_sample = {name: 0 for name in all_binary_vars}

    admm_blocks = []
    for g in range(pd.num_gens):
        admm_blocks.append([f"u_gen{g}_hr{t}" for t in range(pd.num_hours)])
        
    ups_vars_block = [f"ups_total_b{j}" for j in range(J_bits)]
    admm_blocks.append(ups_vars_block)

    parsed_ineqs = parse_cqm_constraints(cqm)
    lambda_vec = np.zeros(len(parsed_ineqs))
    last_R = np.inf
    max_qubits_used = 0
    
    for inner_iter in range(max_inner_iters):
        for block_idx, block in enumerate(admm_blocks):
            bqm = build_qphr_admm_bqm(cqm, parsed_ineqs, lambda_vec, sigma, block, best_sample)
            current_qubits = len(bqm.variables)
            
            if current_qubits > max_qubits_used:
                max_qubits_used = current_qubits
            
            # -------------------------------------------------------------
            # QISKIT QAOA EXECUTION
            # -------------------------------------------------------------
            if current_qubits > 0:
                # 1. Convert to Qiskit Quadratic Program
                qp = bqm_to_qiskit_qp(bqm)
                
                # 2. Run QAOA
                result = qaoa_optimizer.solve(qp)
                
                # 3. Extract Bitstring
                for var, val in zip(qp.variables, result.x):
                    best_sample[var.name] = int(val)
                    
        R = compute_residual_R(parsed_ineqs, best_sample, lambda_vec, sigma)
        
        for i, constr in enumerate(parsed_ineqs):
            g_val = evaluate_g_i(constr, best_sample)
            lambda_vec[i] = max(lambda_vec[i] + sigma * g_val, 0.0)
            
        if R <= 1e-3:
            break
        if R > rho * last_R:
            sigma *= eta
            
        last_R = R

    qubit_history.append(max_qubits_used)

    max_violation = 0.0
    for ineq in parsed_ineqs:
        g_val = evaluate_g_i(ineq, best_sample)
        if g_val > max_violation:
            max_violation = g_val

    master_time = time.perf_counter() - master_start

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

    this_iter_cut = BendersCut(theta=total_theta, rhs=total_rhs, iteration=iteration, scenario=-1)
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

    # ------------------------------------------------------------------
    # Log iterations
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Check convergence
    # ------------------------------------------------------------------
    if np.isfinite(lower_bound) and np.isfinite(upper_bound):
        gap = upper_bound - lower_bound
        tolerance = gap_tolerance * max(1.0, abs(upper_bound))
        
        if gap <= tolerance:
            print_timed(f"Iter {iteration:3d} | LB: {lower_bound:12.2f} | UB: {upper_bound:12.2f} | Gap: {gap:12.2f}")
            print_timed("Converged.")
            break
    
    if max_qubits_used > 1000:
        print_timed(f"Hardware Limit Reached: Qubit count ({max_qubits_used}) exceeds 1000.")
        break

pbar_outer.close()

# ---------------------------------------------------------
# Extract, Format, and Log Results
# ---------------------------------------------------------
import datetime

found_solution = best_u is not None
converged = (upper_bound - lower_bound) <= gap_tolerance * max(1.0, abs(upper_bound))
status_str = "Optimal" if converged else ("Feasible" if found_solution else "Infeasible")
exec_time = sum(i['iteration_time_seconds'] for i in iteration_history)
final_gap = float(upper_bound - lower_bound) 

print(f"\n[3. Results Summary]")
print(f"  -> Solver Status:        {status_str}")
print(f"  -> Found Solution?       {found_solution}")
print(f"  -> Optimal Cost:         {f'${upper_bound:,.2f}' if found_solution else 'N/A'}")
print(f"  -> Final Lower Bound:    {f'${lower_bound:,.2f}'}")
print(f"  -> Final Gap:            {f'${final_gap:,.2f}'}")
print(f"  -> Time Taken:           {exec_time:.2f} seconds")
print(f"  -> Final Qubit Count:    {qubit_history[-1] if qubit_history else 0}")
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
    if spill_cost > 0: print(f"  -> Spilling Penalty:   ${spill_cost:,.2f}")
    print(f"  -> Total Computed:     ${(fixed_cost + expected_recourse_cost):,.2f}")

# --- JSON Export ---
total_bits = pd.num_gens * pd.num_hours

results_payload = {
    "metadata": {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "experiment_name": "Stochastic Unit Commitment - Qiskit QAOA Benders (Single Cut)",
        "solver_name": "Qiskit QAOA Simulator (Benders Master)",
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
        "binary_variables_count": int(total_bits),
        "qubit_history": qubit_history,
        "benders_final_lower_bound": float(lower_bound),
        "benders_final_upper_bound": float(upper_bound),
        "benders_final_gap": float(final_gap),
        "benders_iterations": int(iteration),
        "benders_cuts_generated": int(len(cuts))
    },
    "iteration_history": iteration_history,
    "solution_details": {
        "best_bitstring": best_bitstring,
        "commitment_schedule_by_generator": schedule_by_generator,
        "cost_by_scenario": cost_by_scenario
    }
}

os.makedirs("results", exist_ok=True)
json_path = os.path.join("results", f"qiskit_qaoa_suc_singlecut_benders_{prob_name}.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(results_payload, f, indent=4)

print(f"\n[SUCCESS] Comprehensive JSON logged to: '{json_path}'")