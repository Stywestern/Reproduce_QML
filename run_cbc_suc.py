############################################################################################
#              RUNNER: CLASSICAL PULP CBC SOLVER (24-HOUR STOCHASTIC UC)
############################################################################################

import os
import json
import time
import datetime
import dimod
import pulp
import numpy as np

from src.problems.stochastic_uc_4gen import FourGeneratorUCModel
from src.preprocessing.cqm_to_pulp import cqm_to_pulp


def solve_pulp_cbc(
    num_hours: int = 24,
    num_scenarios: int = 3,
    seed: int = 2
):
    """
    Solves the full 24-hour Stochastic Unit Commitment MILP using PuLP's COIN-OR CBC engine.
    Establishes the exact classical ground-truth optimal baseline cost for benchmarking.
    """
    print("==========================================================================")
    print(f"       RUNNER: CLASSICAL PULP CBC SOLVER ({num_hours}-HOUR SUC)           ")
    print("==========================================================================")

    # 1. Instantiate 24-Hour Problem Model
    print("\n[1. Instantiating Problem Blueprint]")
    problem = FourGeneratorUCModel(num_hours=num_hours, num_scenarios=num_scenarios, seed=seed)
    cqm = problem.get_cqm()

    binary_vars = [v for v in cqm.variables if cqm.vartype(v) == dimod.BINARY]
    real_vars = [v for v in cqm.variables if cqm.vartype(v) == dimod.REAL]
    num_bits = len(binary_vars)
    total_state_space = f"2^{num_bits}"

    print(f"  -> Time Horizon:          {num_hours} hours")
    print(f"  -> Total Decision Vars:   {len(cqm.variables)} (96 Binary Qubits + 96 Real MW Outputs)")
    print(f"  -> Total Constraints:     {len(cqm.constraints)}")
    print(f"  -> Search Space:          {total_state_space} states")

    # 2. Translate CQM to PuLP MILP
    print("\n[2. Preprocessing: Converting CQM -> PuLP LpProblem]")
    pulp_prob, pulp_vars = cqm_to_pulp(cqm)

    # 3. Execute PuLP CBC Branch-and-Cut Engine
    print("\n[3. Executing COIN-OR CBC Branch-and-Cut Solver]")
    solver = pulp.PULP_CBC_CMD(msg=True)  # Set msg=True if you want raw C++ solver stdout

    start_time = time.perf_counter()
    status_code = pulp_prob.solve(solver)
    end_time = time.perf_counter()

    exec_time = round(end_time - start_time, 4)
    status_str = pulp.LpStatus[status_code]

    # 4. Extract Solution Details & Evaluate Convergence
    found_solution = bool(status_str in ["Optimal", "Feasible"])
    converged = bool(status_str == "Optimal")  # Branch-and-Bound proves true global optimality
    
    cost_of_solution = float(round(pulp.value(pulp_prob.objective), 2)) if found_solution else None

    print(f"\n[4. Results Summary]")
    print(f"  -> Solver Status:        {status_str}")
    print(f"  -> Found Solution?       {found_solution}")
    print(f"  -> Optimal Cost:         ${cost_of_solution:,.2f}" if found_solution else "  -> Optimal Cost:         N/A")
    print(f"  -> Time Taken:           {exec_time} seconds")
    print(f"  -> Algorithm Converged?  {converged}")

    # 5. Extract Hour-by-Hour Generator Commitment Schedule
    schedule_by_generator = {gen_name: [] for gen_name in problem.gen_names}
    best_bitstring = ""

    if found_solution:
        # Construct bitstring and schedule matrix from binary u_{gen}_t{t} decisions
        u_matrix = np.zeros((problem.num_gens, num_hours), dtype=int)
        for t in range(num_hours):
            for i, gen_name in enumerate(problem.gen_names):
                var_key = f"u_{gen_name}_t{t}"
                val = int(round(pulp_vars[var_key].varValue)) if pulp_vars[var_key].varValue is not None else 0
                u_matrix[i, t] = val
                schedule_by_generator[gen_name].append(val)

        best_bitstring = "".join(u_matrix.flatten().astype(str))

    # 6. Build Comprehensive JSON Payload (Matching standard schema)
    results_payload = {
        "metadata": {
            "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "experiment_name": "4-Generator Stochastic Unit Commitment",
            "solver_name": "PuLP CBC (Classical Ground Truth)",
            "seed_used": int(seed)
        },
        "core_required_metrics": {
            "found_solution": bool(found_solution),
            "cost_of_solution": cost_of_solution,
            "time_it_took_seconds": float(exec_time),
            "converged": bool(converged)
        },
        "extended_solver_metrics": {
            "solver_status": str(status_str),
            "num_hours": int(num_hours),
            "num_scenarios": int(num_scenarios),
            "total_variables_count": int(len(cqm.variables)),
            "binary_variables_count": int(num_bits),
            "continuous_variables_count": int(len(real_vars)),
            "total_constraints_count": int(len(cqm.constraints)),
            "total_search_space_formula": str(total_state_space)
        },
        "solution_details": {
            "best_bitstring": best_bitstring,
            "commitment_schedule_by_generator": schedule_by_generator
        }
    }

    # Save to /results directory
    os.makedirs("results", exist_ok=True)
    json_path = os.path.join("results", "pulp_cbc_suc.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=4)

    print(f"\n[SUCCESS] Comprehensive JSON logged to: '{json_path}'")
    print("==========================================================================\n")


if __name__ == "__main__":
    solve_pulp_cbc(
        num_hours=24,
        num_scenarios=3,
        seed=2
    )