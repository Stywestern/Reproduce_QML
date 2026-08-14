import os
import json
import time
import datetime
import pulp
import numpy as np

# Assuming lbo_to_pulp is defined above or imported
from src.problems.suc_problem_generator import SUCProblemData, SUCModelBuilder, MixedLBOProblem

def lbo_to_pulp(model: MixedLBOProblem) -> tuple[pulp.LpProblem, dict[str, pulp.LpVariable]]:
    """
    Translates a generic MixedLBOProblem into a PuLP model using sparse matrix iteration.
    """
    prob = pulp.LpProblem("SUC_Master_Problem", pulp.LpMinimize)
    pulp_vars = {}
    pulp_vars_list = []

    # 1. Initialize Variables
    for i, name in enumerate(model.var_names):
        vtype = pulp.LpBinary if model.var_types[i] == "B" else pulp.LpContinuous
        
        # Handle bounds. PuLP uses None for infinity.
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
    # A is a scipy.sparse.csr_matrix. We iterate over its rows efficiently.
    for i in range(model.A.shape[0]):
        row_start = model.A.indptr[i]
        row_end = model.A.indptr[i+1]
        
        if row_start == row_end:
            continue # Empty row
            
        # Construct the linear expression for this row
        expr = pulp.lpSum(
            float(model.A.data[j]) * pulp_vars_list[model.A.indices[j]] 
            for j in range(row_start, row_end)
        )
        
        lower_bound = model.row_lower[i]
        upper_bound = model.row_upper[i]
        label = model.constraint_labels[i]

        # Apply bounds
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
    num_hours = 24
    num_scenarios = 3
    seed = 2

    print("==========================================================================")
    print(f"       RUNNER: CLASSICAL PULP CBC SOLVER ({num_hours}-HOUR SUC)          ")
    print("==========================================================================")

    # ---------------------------------------------------------
    # 1. Instantiate Problem Blueprint
    # ---------------------------------------------------------
    print("\n[1. Instantiating Problem Blueprint]")
    rng = np.random.default_rng(seed)
    t = np.arange(num_hours)
    
    # Base demand swings smoothly from ~130 MW (night) to ~310 MW (day peak)
    base_demand = 220 + 90 * np.sin((t - 6) * np.pi / 12)
    custom_demand = np.array([
        base_demand + rng.normal(0, 10, num_hours) 
        for _ in range(num_scenarios)
    ])
    
    # Wind fluctuates between 0 and 40 MW
    custom_wind = np.array([
        np.clip(rng.normal(20, 15, num_hours), 0, 40) 
        for _ in range(num_scenarios)
    ])

    pd = SUCProblemData.generate(
        num_gens=4, 
        num_hours=num_hours, 
        num_scenarios=num_scenarios, 
        seed=seed,
        p_max=np.array([100.0, 95.0, 80.0, 75.0]),  # 350 MW Total
        p_min=np.array([40.0, 35.0, 30.0, 25.0]),   # 130 MW Total Min
        c_var=np.array([25.0, 26.0, 27.0, 28.0]),
        c_fixed=np.array([300.0, 290.0, 280.0, 270.0]),
        min_up=np.array([10, 8, 8, 6]),
        min_down=np.array([10, 8, 8, 6]),
        demand=custom_demand,
        wind=custom_wind
    )

    builder = SUCModelBuilder(pd)
    builder.get_base_model()                # 1. Populates the constraints in the builder
    base_model = builder.builder.compile()  # 2. Compiles them into the MixedLBOProblem matrices

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
    print("\n[2. Preprocessing: Converting MixedLBOProblem -> PuLP LpProblem]")
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
    found_solution = bool(status_str in ["Optimal", "Feasible"])
    converged = bool(status_str == "Optimal")
    
    cost_of_solution = float(round(pulp.value(pulp_prob.objective), 2)) if found_solution else None

    print(f"\n[4. Results Summary]")
    print(f"  -> Solver Status:        {status_str}")
    print(f"  -> Found Solution?       {found_solution}")
    if found_solution:
        print(f"  -> Optimal Cost:         ${cost_of_solution:,.2f}")
    else:
        print(f"  -> Optimal Cost:         N/A")
    print(f"  -> Time Taken:           {exec_time} seconds")
    print(f"  -> Algorithm Converged?  {converged}")

    schedule_by_generator = {f"gen{g}": [] for g in range(pd.num_gens)}
    best_bitstring = ""

    if found_solution:
        # Extract Hour-by-Hour Generator Commitment Schedule
        u_matrix = np.zeros((pd.num_gens, num_hours), dtype=int)
        for t in range(num_hours):
            for g in range(pd.num_gens):
                var_key = f"u_gen{g}_hr{t}"
                val_raw = pulp_vars[var_key].varValue
                val = int(round(val_raw)) if val_raw is not None else 0
                u_matrix[g, t] = val
                schedule_by_generator[f"gen{g}"].append(val)

        best_bitstring = "".join(u_matrix.flatten().astype(str))

        # --- NEW COST BREAKDOWN SECTION ---
        fixed_cost = 0.0
        dispatch_cost = 0.0
        shed_cost = 0.0
        
        for i, name in enumerate(base_model.var_names):
            coeff = float(base_model.c[i])
            if coeff == 0: 
                continue
            
            val = pulp_vars[name].varValue
            if val is None: val = 0.0
            
            term_cost = coeff * val
            if name.startswith("u_gen"):
                fixed_cost += term_cost
            elif name.startswith("P_gen"):
                dispatch_cost += term_cost
            elif name.startswith("shed_"):
                shed_cost += term_cost

        print(f"\n[5. Objective Cost Breakdown]")
        print(f"  -> Fixed Commitment Costs:   ${fixed_cost:,.2f}")
        print(f"  -> Expected Dispatch Costs:  ${dispatch_cost:,.2f}")
        print(f"  -> Expected Shedding Penalty:${shed_cost:,.2f}")
        print("-" * 50)
        print(f"  -> Total Computed Cost:      ${(fixed_cost + dispatch_cost + shed_cost):,.2f}")

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
            "commitment_schedule_by_generator": schedule_by_generator
        }
    }

    os.makedirs("results", exist_ok=True)
    json_path = os.path.join("results", "pulp_cbc_suc.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=4)

    print(f"\n[SUCCESS] Comprehensive JSON logged to: '{json_path}'")
    print("==========================================================================\n")