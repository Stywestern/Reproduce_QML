############################################################################################
#       RUNNER: CHUNKED 6-HOUR ADMM + SIMULATED ANNEALING (24-HOUR STOCHASTIC UC)
############################################################################################

import os
import json
import time
import datetime
import warnings
import numpy as np
import pulp
import dimod
import neal
from tqdm import tqdm

# Custom Modules
from src.problems.stochastic_uc_4gen import FourGeneratorUCModel
from src.preprocessing.cqm_to_bqm import cqm_to_bqm_slack

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def solve_continuous_dispatch(u_matrix, problem: FourGeneratorUCModel) -> float:
    """STAGE 2: Continuous LP Dispatcher to calculate exact fuel costs."""
    lp = pulp.LpProblem("Economic_Dispatch", pulp.LpMinimize)
    P = {}
    for t in range(problem.num_hours):
        for i in range(problem.num_gens):
            is_on = int(u_matrix[i, t]) == 1
            lb = float(problem.p_min[i]) if is_on else 0.0
            ub = float(problem.p_max[i]) if is_on else 0.0
            P[i, t] = pulp.LpVariable(f"P_gen{i}_t{t}", lowBound=lb, upBound=ub, cat=pulp.LpContinuous)

    var_cost_terms = [problem.c_var[i] * P[i, t] for t in range(problem.num_hours) for i in range(problem.num_gens)]
    lp += pulp.lpSum(var_cost_terms)

    for t in range(problem.num_hours):
        for s in range(problem.num_scenarios):
            net_demand = problem.demand[s, t] - problem.wind[s, t]
            lp += pulp.lpSum(P[i, t] for i in range(problem.num_gens)) >= net_demand

    lp.solve(pulp.PULP_CBC_CMD(msg=False))
    return float(pulp.value(lp.objective)) if pulp.LpStatus[lp.status] == 'Optimal' else float('inf')


def build_chunk_cqm(problem: FourGeneratorUCModel, start_t: int, block_size: int, prev_u_matrix: np.ndarray, lambdas: np.ndarray, rho: float):
    """
    Builds a standalone CQM for a MULTI-HOUR CHUNK (e.g., 6 hours at once).
    This gives the annealer local foresight across hours and thermal inertia.
    """
    cqm = dimod.ConstrainedQuadraticModel()
    end_t = min(start_t + block_size, problem.num_hours)
    
    u = {}
    for t in range(start_t, end_t):
        for i in range(problem.num_gens):
            u[i, t] = dimod.Binary(f"u_gen{i}_t{t}")

    # Objective: Fixed Costs + ADMM Penalties linking to neighboring hours
    obj = 0
    for t in range(start_t, end_t):
        for i in range(problem.num_gens):
            obj += problem.c_fixed[i] * u[i, t]
            
            # Link to previous hour if it crosses outside this block or within
            if t > 0 and prev_u_matrix is not None:
                u_prev = float(prev_u_matrix[i, t-1])
                obj += float(lambdas[i, t]) * (u[i, t] - u_prev)
                obj += (rho / 2.0) * (u[i, t] - 2.0 * u_prev * u[i, t] + (u_prev ** 2))

    cqm.set_objective(obj)

    # Local Capacity Constraints for each hour in the chunk
    for t in range(start_t, end_t):
        for s in range(problem.num_scenarios):
            net_demand = problem.demand[s, t] - problem.wind[s, t]
            cap_expr = sum(problem.p_max[i] * u[i, t] for i in range(problem.num_gens))
            cqm.add_constraint(cap_expr >= net_demand, label=f"cap_s{s}_t{t}")

    var_names = [f"u_gen{i}_t{t}" for t in range(start_t, end_t) for i in range(problem.num_gens)]
    return cqm, var_names


def run_hybrid_admm_sa(num_hours: int = 24, block_size: int = 6, max_admm_iters: int = 20, seed: int = 2):
    """
    Executes ADMM using multi-hour chunk blocks (e.g., 4 blocks of 6 hours) 
    to drastically improve quantum foresight and convergence stability.
    """
    print("==========================================================================")
    print(f"    RUNNER: CHUNKED ({block_size}-HR BLOCK) ADMM + SA ({num_hours}-HOUR SUC) ")
    print("==========================================================================")

    problem = FourGeneratorUCModel(num_hours=num_hours, num_scenarios=3, seed=seed)
    
    rho = 100.0
    lambdas = np.zeros((problem.num_gens, num_hours))
    
    # Warm-start initialization
    current_u_matrix = np.zeros((problem.num_gens, num_hours), dtype=int)
    for t in range(num_hours):
        current_u_matrix[0, t] = 1  # Base Coal on
        net_peak = problem.demand[0, t] - problem.wind[0, t]
        if net_peak > 150.0: current_u_matrix[1, t] = 1
        if net_peak > 220.0: current_u_matrix[2, t] = 1

    sampler = neal.SimulatedAnnealingSampler()

    best_cost = float('inf')
    best_schedule = None
    total_annealer_calls = 0
    start_time = time.perf_counter()


    num_blocks = int(np.ceil(num_hours / block_size))
    converged = False
    iteration = 0

    # --- MAIN ADMM PROGRESS BAR (Wraps the overall run) ---
    with tqdm(total=max_admm_iters, desc="ADMM Convergence Progress", unit="iter", dynamic_ncols=True) as pbar:
        while iteration < max_admm_iters and not converged:
            iteration += 1
            pbar.set_description(f"ADMM Iter {iteration} (Rho: {rho:,.1f})")
            
            previous_u_matrix = current_u_matrix.copy()
            
            # --- INNER LOOP: Solve Multi-Hour Blocks ---
            for b in range(num_blocks):
                start_t = b * block_size
                end_t = min(start_t + block_size, num_hours)
                
                cqm, original_vars = build_chunk_cqm(problem, start_t, block_size, current_u_matrix, lambdas, rho)
                bqm, invert_map, q_vars, num_slacks = cqm_to_bqm_slack(cqm, lagrange_mult=3000.0)
                
                sampleset = sampler.sample(
                    bqm, 
                    num_reads=800, 
                    num_sweeps=5000, 
                    beta_schedule_type="geometric", 
                    seed=seed + iteration + b
                )
                total_annealer_calls += 1
                
                best_sample = sampleset.first.sample
                for var_name in original_vars:
                    parts = var_name.split('_')
                    gen_idx = int(parts[1].replace('gen', ''))
                    t_val = int(parts[2].replace('t', ''))
                    current_u_matrix[gen_idx, t_val] = int(best_sample.get(var_name, 0))

            # --- CLASSICAL STAGE 2: Evaluate Schedule ---
            fixed_cost = float(np.sum(current_u_matrix.T * problem.c_fixed))
            variable_cost = solve_continuous_dispatch(current_u_matrix, problem)
            
            is_feasible = variable_cost != float('inf')
            total_real_cost = fixed_cost + variable_cost if is_feasible else float('inf')
            
            if is_feasible:
                if total_real_cost < best_cost:
                    best_cost = total_real_cost
                    best_schedule = current_u_matrix.copy()
                    tqdm.write(f"  [Iter {iteration}] Valid Schedule | Current Cost: ${total_real_cost:,.2f} | [*] NEW BEST!")
                else:
                    tqdm.write(f"  [Iter {iteration}] Valid Schedule | Current Cost: ${total_real_cost:,.2f} (Best: ${best_cost:,.2f})")
            else:
                tqdm.write(f"  [Iter {iteration}] Invalid Schedule Generated. Retaining previous state...")
                current_u_matrix = previous_u_matrix.copy()

            # --- CHECK CONVERGENCE ---
            schedule_diff = np.max(np.abs(current_u_matrix - previous_u_matrix))
            if schedule_diff == 0:
                tqdm.write(f"  [CONVERGENCE REACHED] Binary schedule stabilized perfectly at iteration {iteration}!")
                converged = True
                pbar.update(1)
                break
            else:
                tqdm.write(f"  -> Schedule delta: {schedule_diff}")

            # --- ADMM PENALTY UPDATE ---
            for t in range(1, num_hours):
                for i in range(problem.num_gens):
                    gap = float(current_u_matrix[i, t] - current_u_matrix[i, t-1])
                    lambdas[i, t] += rho * gap
            
            rho = min(rho * 1.05, 5000.0)
            pbar.update(1)

    end_time = time.perf_counter()
    exec_time = round(end_time - start_time, 4)

    print(f"\n[SUCCESS] Loop finished. Converged: {bool(converged)} after {iteration} iterations.")
    print(f"[SUCCESS] Best Real Cost Found: ${best_cost:,.2f}")

    if best_schedule is None:
        best_schedule = current_u_matrix

    found_sol = bool(best_cost != float('inf'))
    cost_val = float(round(best_cost, 2)) if found_sol else None

    results_payload = {
        "metadata": {
            "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "experiment_name": f"{num_hours}-Hour Chunked SUC ({block_size}-hour blocks)",
            "solver_name": "Chunked Hybrid ADMM + Simulated Annealing (Neal)",
            "seed_used": int(seed)
        },
        "core_required_metrics": {
            "found_solution": bool(found_sol),
            "cost_of_solution": cost_val,
            "time_it_took_seconds": float(exec_time),
            "converged": bool(converged)
        },
        "extended_solver_metrics": {
            "num_hours": int(num_hours),
            "block_size_hours": int(block_size),
            "admm_iterations": int(iteration),
            "total_annealer_subproblems_solved": int(total_annealer_calls)
        },
        "solution_details": {
            "commitment_schedule_by_generator": {
                str(problem.gen_names[i]): [int(val) for val in best_schedule[i, :]] 
                for i in range(problem.num_gens)
            }
        }
    }

    os.makedirs("results", exist_ok=True)
    json_path = os.path.join("results", "admm_chunked_sa_suc.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=4)

    print(f"\n[SUCCESS] JSON logged to: '{json_path}'")
    print("==========================================================================\n")

if __name__ == "__main__":
    run_hybrid_admm_sa(num_hours=24, block_size=6, max_admm_iters=15, seed=2)