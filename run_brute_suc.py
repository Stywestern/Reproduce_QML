############################################################################################
#              RUNNER: BRUTE-FORCE / RANDOM SAMPLING SEARCH (24-HOUR SUC)
############################################################################################

import os
import json
import time
import datetime
import numpy as np
from tqdm import tqdm

from src.problems.stochastic_uc_4gen import FourGeneratorUCModel


def solve_brute_force_sampling(
    num_hours: int = 24,
    num_scenarios: int = 3,
    sample_budget: int = 1000000,
    batch_size: int = 1000,
    seed: int = 2
):
    """
    Executes a random sampling search over the 24-hour Stochastic Unit Commitment space (2^96 states).
    Evaluates 'sample_budget' random binary configurations using vectorized NumPy batches & tqdm monitoring.
    """
    print("==========================================================================")
    print(f"       RUNNER: BRUTE-FORCE RANDOM SAMPLING ({num_hours}-HOUR SUC)          ")
    print("==========================================================================")

    # 1. Instantiate 24-Hour Problem Model
    problem = FourGeneratorUCModel(num_hours=num_hours, num_scenarios=num_scenarios, seed=seed)
    
    num_gens = problem.num_gens
    num_bits = num_gens * num_hours
    total_state_space = f"2^{num_bits}"

    print(f"\n[1. Problem Dimensions]")
    print(f"  -> Time Horizon:          {num_hours} hours")
    print(f"  -> Binary Qubits:         {num_bits} variables")
    print(f"  -> Total Search Space:    {total_state_space}")
    print(f"  -> Random Sample Budget:  {sample_budget:,} states (Batch size: {batch_size:,})")

    # Set random seed for reproducible sampling
    np.random.seed(seed)

    best_cost = float("inf")
    best_commitment_matrix = None
    best_bitstring = None
    feasible_samples_count = 0

    start_time = time.perf_counter()

    # 2. Vectorized Batch Sampling with tqdm Progress Bar
    num_batches = int(np.ceil(sample_budget / batch_size))
    
    pbar = tqdm(total=sample_budget, desc="Evaluating States", unit="sample", dynamic_ncols=True)

    for b in range(num_batches):
        current_batch_size = min(batch_size, sample_budget - b * batch_size)
        
        # Generate batch of shape (current_batch_size, num_gens, num_hours)
        batch_u = np.random.randint(0, 2, size=(current_batch_size, num_gens, num_hours))

        for k in range(current_batch_size):
            u_matrix = batch_u[k]

            # Check Feasibility across all hours & scenarios
            is_feasible = True
            for t in range(num_hours):
                for s in range(num_scenarios):
                    net_demand = problem.demand[s, t] - problem.wind[s, t]
                    online_cap = np.sum(problem.p_max * u_matrix[:, t])
                    
                    if online_cap < net_demand:
                        is_feasible = False
                        break
                if not is_feasible:
                    break

            # Evaluate Cost if Feasible
            if is_feasible:
                feasible_samples_count += 1

                # Compute Fixed Commitment Cost
                fixed_cost = 0.0
                for t in range(num_hours):
                    for i in range(num_gens):
                        fixed_cost += problem.c_fixed[i] * u_matrix[i, t]

                if fixed_cost < best_cost:
                    best_cost = fixed_cost
                    best_commitment_matrix = u_matrix.copy()
                    best_bitstring = "".join(u_matrix.flatten().astype(str))

        # Update tqdm progress bar
        pbar.update(current_batch_size)
        best_cost_str = f"${best_cost:,.2f}" if best_cost < float("inf") else "None"
        pbar.set_postfix({
            "Feasible": f"{feasible_samples_count:,}",
            "Best Cost": best_cost_str
        })

    pbar.close()

    end_time = time.perf_counter()
    exec_time = round(end_time - start_time, 4)

    # 3. Derive Core Standardized Metrics (Explicit Native Python Casting)
    found_solution = bool(best_cost < float("inf"))
    cost_of_solution = float(round(best_cost, 2)) if found_solution else None
    converged = False

    print(f"\n[2. Results Summary]")
    print(f"  -> Found Solution?       {found_solution}")
    print(f"  -> Cost of Solution:     ${cost_of_solution:,.2f}" if found_solution else "  -> Cost of Solution:     N/A (No Feasible State Found)")
    print(f"  -> Time Taken:           {exec_time} seconds")
    print(f"  -> Algorithm Converged?  {converged}")
    print(f"  -> Feasible Samples:     {feasible_samples_count:,} / {sample_budget:,} ({feasible_samples_count/sample_budget*100:.5f}%)")

    # 4. Format Hour-by-Hour Schedule for Rich Output Logging
    schedule_by_generator = {}
    if found_solution and best_commitment_matrix is not None:
        for i, gen_name in enumerate(problem.gen_names):
            # Convert NumPy array elements to native Python ints for JSON serialization
            schedule_by_generator[gen_name] = [int(val) for val in best_commitment_matrix[i, :]]

    # 5. Build Comprehensive JSON Payload (All Native Python Types)
    results_payload = {
        "metadata": {
            "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "experiment_name": "4-Generator Stochastic Unit Commitment",
            "solver_name": "Brute-Force / Random Sampling Baseline",
            "seed_used": int(seed)
        },
        "core_required_metrics": {
            "found_solution": bool(found_solution),
            "cost_of_solution": cost_of_solution,
            "time_it_took_seconds": float(exec_time),
            "converged": bool(converged)
        },
        "extended_solver_metrics": {
            "num_hours": int(num_hours),
            "num_scenarios": int(num_scenarios),
            "binary_variables_count": int(num_bits),
            "total_search_space_formula": str(total_state_space),
            "samples_evaluated": int(sample_budget),
            "feasible_samples_count": int(feasible_samples_count),
            "feasibility_rate_percent": float(round(feasible_samples_count / sample_budget * 100, 5)),
            "average_time_per_sample_us": float(round((exec_time / sample_budget) * 1e6, 4))
        },
        "solution_details": {
            "best_bitstring": best_bitstring,
            "commitment_schedule_by_generator": schedule_by_generator
        }
    }

    # Save to /results directory
    os.makedirs("results", exist_ok=True)
    json_path = os.path.join("results", "brute_force_suc.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=4)

    print(f"\n[SUCCESS] Comprehensive JSON logged to: '{json_path}'")
    print("==========================================================================\n")


if __name__ == "__main__":
    solve_brute_force_sampling(
        num_hours=24,
        num_scenarios=3,
        sample_budget=10**8,
        batch_size=10000,
        seed=2
    )