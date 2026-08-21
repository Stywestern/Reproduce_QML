import os
import json
import time
import datetime
import pulp
import numpy as np

from src.problems.suc_problem_generator import SUCProblemData, SUCModelBuilder, MILPModel
from src.problems.generated_problems import g4t24s3_suc_problem, g10t24s10_suc_problem, tiny_suc_problem, long_horizon_suc_problem, many_scenario_suc_problem

def lbo_to_pulp(model: MILPModel) -> tuple[pulp.LpProblem, dict[str, pulp.LpVariable]]:
    """
    Translates a MILPModel into a PuLP model using sparse matrix iteration.
    """
    prob = pulp.LpProblem("SUC_Master_Problem", pulp.LpMinimize)
    pulp_vars = {}
    pulp_vars_list = []

    # 1. Initialize Variables
    for i, name in enumerate(model.var_names):
        vtype = pulp.LpBinary if model.var_types[i] == "B" else pulp.LpContinuous

        lb = None if model.lower[i] == -np.inf else float(model.lower[i])
        ub = None if model.upper[i] == np.inf else float(model.upper[i])

        var = pulp.LpVariable(name, lowBound=lb, upBound=ub, cat=vtype)
        pulp_vars[name] = var
        pulp_vars_list.append(var)

    # 2. Build Objective Function (c^T x)
    prob += pulp.lpSum(
        float(model.c[i]) * pulp_vars_list[i]
        for i in range(len(model.c)) if model.c[i] != 0
    )

    # 3. Build Constraints from Sparse Matrix A
    for i in range(model.A.shape[0]):
        row_start = model.A.indptr[i]
        row_end = model.A.indptr[i + 1]

        if row_start == row_end:
            continue  # Empty row

        expr = pulp.lpSum(
            float(model.A.data[j]) * pulp_vars_list[model.A.indices[j]]
            for j in range(row_start, row_end)
        )

        lower_bound = model.row_lower[i]
        upper_bound = model.row_upper[i]
        label = model.constraint_labels[i]

        if lower_bound == upper_bound:
            prob += (expr == float(upper_bound), label)
        else:
            if lower_bound > -np.inf:
                prob += (expr >= float(lower_bound), f"{label}_lb")
            if upper_bound < np.inf:
                prob += (expr <= float(upper_bound), f"{label}_ub")

    return prob, pulp_vars


if __name__ == "__main__":
    # Configuration

    # ---------------------------------------------------------
    # 1. Instantiate Problem
    # ---------------------------------------------------------
    print("\n[1. Instantiating Problem Blueprint]")
    seed = 2
    num_hours = 24
    num_scenarios = 2

    #pd, prob_name = tiny_suc_problem(seed=seed)
    #pd, prob_name = long_horizon_suc_problem(seed=seed)
    pd, prob_name = g4t24s3_suc_problem(seed=seed)
    #pd, prob_name = g10t24s10_suc_problem(seed=seed)

    builder = SUCModelBuilder(pd)
    base_model = builder.get_base_model()  # MILPModel is already the compiled matrix form

    binary_vars = [v for v in base_model.var_types if v == "B"]
    real_vars = [v for v in base_model.var_types if v == "C"]
    num_bits = len(binary_vars)
    total_state_space = f"2^{num_bits}"

    print(f"  -> Time Horizon:          {num_hours} hours")
    print(f"  -> Total Decision Vars:   {len(base_model.var_names)} ({num_bits} Binary + {len(real_vars)} Continuous)")
    print(f"  -> Total Constraints:     {len(base_model.constraint_labels)}")
    print(f"  -> Search Space:          {total_state_space} states")

    # ---------------------------------------------------------
    # 2. Translate to PuLP
    # ---------------------------------------------------------
    print("\n[2. Preprocessing: Converting MILPModel -> PuLP LpProblem]")
    pulp_prob, pulp_vars = lbo_to_pulp(base_model)

    # ---------------------------------------------------------
    # 3. Execute Solver
    # ---------------------------------------------------------
    print("\n[3. Executing COIN-OR CBC Branch-and-Cut Solver]")
    solver = pulp.PULP_CBC_CMD(msg=True)

    start_time = time.perf_counter()
    status_code = pulp_prob.solve(solver)
    end_time = time.perf_counter()

    exec_time = round(end_time - start_time, 4)
    status_str = pulp.LpStatus[status_code]

# ---------------------------------------------------------
# 4. Extract and Log Results
# ---------------------------------------------------------
found_solution = status_str in ["Optimal", "Feasible"]
converged = status_str == "Optimal"
cost_of_solution = float(round(pulp.value(pulp_prob.objective), 2)) if found_solution else None

print(f"\n[4. Results Summary]")
print(f"  -> Solver Status:       {status_str}")
print(f"  -> Found Solution?       {found_solution}")
print(f"  -> Optimal Cost:         {f'${cost_of_solution:,.2f}' if found_solution else 'N/A'}")
print(f"  -> Time Taken:           {exec_time} seconds")
print(f"  -> Algorithm Converged?  {converged}")

schedule_by_generator = {f"gen{g}": [] for g in range(pd.num_gens)}
best_bitstring = ""
cost_by_scenario = {}
fixed_cost = dispatch_cost = shed_cost = 0.0
expected_recourse_cost = 0.0

if found_solution:
    # --- Extract Commitment Schedule ---
    u_matrix = np.zeros((pd.num_gens, num_hours), dtype=int)
    for t in range(num_hours):
        for g in range(pd.num_gens):
            val = int(round(pulp_vars[f"u_gen{g}_hr{t}"].varValue or 0))
            u_matrix[g, t] = val
            schedule_by_generator[f"gen{g}"].append(val)
    best_bitstring = "".join(u_matrix.flatten().astype(str))

    # --- Cost Breakdown by Term Type ---
    for i, name in enumerate(base_model.var_names):
        coeff, val = float(base_model.c[i]), pulp_vars[name].varValue or 0.0
        term_cost = coeff * val
        if name.startswith("u_gen"): fixed_cost += term_cost
        elif name.startswith("P_gen"): dispatch_cost += term_cost
        elif name.startswith("shed_"): shed_cost += term_cost

    print(f"\n[5. Cost Breakdown]")
    print(f"  -> Fixed Costs:        ${fixed_cost:,.2f}")
    print(f"  -> Dispatch Costs:     ${dispatch_cost:,.2f}")
    print(f"  -> Shedding Penalty:   ${shed_cost:,.2f}")
    print(f"  -> Total Computed:     ${(fixed_cost + dispatch_cost + shed_cost):,.2f}")

    # --- Cost Breakdown by Scenario ---
    print(f"\n[6. Cost by Scenario]")
    for s in range(num_scenarios):
        weighted_cost = sum(
            float(base_model.c[i]) * (pulp_vars[base_model.var_names[i]].varValue or 0.0)
            for i in base_model.scenario_vars(s)
        )
        prob = pd.scenario_probs[s]
        raw_cost = weighted_cost / prob if prob > 0 else 0.0
        cost_by_scenario[f"scenario_{s}"] = round(raw_cost, 2)
        expected_recourse_cost += weighted_cost

        print(f"  -> Scenario {s}: P={prob:.3f}, Raw=${raw_cost:,.2f}, Weighted=${weighted_cost:,.2f}")

    print("-" * 50)
    print(f"  -> Expected Total Cost: ${fixed_cost + expected_recourse_cost:,.2f}")

# --- JSON Export ---
results_payload = {
    "metadata": {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "experiment_name": "4-Generator Stochastic Unit Commitment",
        "solver_name": "PuLP CBC (Classical Ground Truth)",
        "seed_used": int(seed)
    },
    "core_required_metrics": {
        "found_solution": found_solution,
        "cost_of_solution": cost_of_solution,
        "time_it_took_seconds": float(exec_time),
        "converged": converged
    },
    "extended_solver_metrics": {
        "solver_status": status_str,
        "num_hours": int(num_hours),
        "num_scenarios": int(num_scenarios),
        "total_variables_count": int(len(base_model.var_names)),
        "binary_variables_count": int(num_bits),
        "continuous_variables_count": int(len(real_vars)),
        "total_constraints_count": int(len(base_model.constraint_labels)),
        "total_search_space_formula": total_state_space
    },
    "solution_details": {
        "best_bitstring": best_bitstring,
        "commitment_schedule_by_generator": schedule_by_generator,
        "cost_by_scenario": cost_by_scenario
    }
}

os.makedirs("results", exist_ok=True)
json_path = os.path.join("results", f"pulp_cbc_suc_{prob_name}.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(results_payload, f, indent=4)

print(f"\n[SUCCESS] Comprehensive JSON logged to: '{json_path}'")
print("==========================================================================\n")