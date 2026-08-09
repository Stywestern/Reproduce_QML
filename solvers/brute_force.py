############################################################################################
#                                           IMPORTS 
############################################################################################

import itertools
import time
import dimod

############################################################################################
#                                           Main Code
############################################################################################

def solve_brute_force(cqm: dimod.ConstrainedQuadraticModel):
    """
    A brute-force solver that checks all possible binary combinations 
    for a given Constrained Quadratic Model (CQM).
    Returns the best cost, the best solutions, and performance metrics.
    """

    print("--- [Brute Force Solver] Initiated ---")
    
    # Extract variables
    variables = list(cqm.variables)
    num_vars = len(variables)
    
    for var in variables:
        if cqm.vartype(var) != dimod.BINARY:
            raise ValueError(f"Variable {var} is not BINARY. Brute force aborted.")
            
    best_cost = float('inf')
    best_sols = []
    
    # --- Telemetry Setup ---
    atomic_calculations = 0
    start_time = time.perf_counter()
    
    total_combinations = 2 ** num_vars
    print(f"[Brute Force Solver] Evaluating all {total_combinations} possible combinations...")
    
    for vals in itertools.product([0, 1], repeat=num_vars):
        # 1. Atomic Calculation: Evaluating one specific combination
        atomic_calculations += 1 
        
        sample = dict(zip(variables, vals))
        
        # Check feasibility and calculate cost
        if cqm.check_feasible(sample):
            cost = cqm.objective.energy(sample)
            
            if cost < best_cost:
                best_cost = cost
                best_sols = [sample]
            elif cost == best_cost:
                best_sols.append(sample)
                
    # --- Telemetry Finalization ---
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    
    # Last Answer's Validity
    if best_sols:
        validity = f"Exact Global Optimum Verified (Cost: {best_cost})"
    else:
        validity = "Infeasible (No valid solutions exist within constraints)"
        
    # --- New Paradigm-Specific Metrics Schema ---
    metrics = {
        "execution_time_seconds": execution_time,
        "validity": validity,
        "paradigm": "Classical Brute Force",
        "paradigm_details": {
            "combinations_evaluated": atomic_calculations
        }
    }
    
    print(f"[Brute Force Solver] Search complete.")
    print(f"  -> Paradigm:     {metrics['paradigm']}")
    print(f"  -> Combinations: {metrics['paradigm_details']['combinations_evaluated']}")
    print(f"  -> Time:         {metrics['execution_time_seconds']:.6f} seconds")
    print(f"  -> Validity:     {metrics['validity']}\n")
    
    return best_cost, best_sols, metrics

############################################################################################
#                               Execution Block
############################################################################################
if __name__ == "__main__":
    print("--- RUNNING STANDALONE SOLVER TEST ---")
    
    # Create a tiny dummy CQM just to test the solver independently
    print("[1] Creating a dummy 3-variable problem...")
    test_cqm = dimod.ConstrainedQuadraticModel()
    x = [dimod.Binary(f"x{i}") for i in range(3)]
    
    # Objective: Minimize x0 - x1 + 2*x2
    test_cqm.set_objective(x[0] - x[1] + 2*x[2])
    # Constraint: x0 + x1 == 1 (Only one can be true)
    test_cqm.add_constraint(x[0] + x[1] == 1, label='c1')
    
    print("[2] Passing dummy model to the solver...\n")
    best_cost, best_solutions, metrics = solve_brute_force(test_cqm)
    
    print(">>> FINAL TEST RESULTS <<<")
    print(f"Minimum Cost: {best_cost}")
    print(f"Optimal Solutions Found: {len(best_solutions)}")
    for idx, sol in enumerate(best_solutions, 1):
        print(f"  Sol {idx}: {sol}")