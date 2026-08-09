import time
import dimod
import pulp
import os
import re

def solve_classical_optimized(cqm: dimod.ConstrainedQuadraticModel):
    """
    An optimized classical solver using PuLP (Branch-and-Bound).
    Dynamically translates a linear D-Wave CQM into a PuLP LP problem.
    """
    print("--- [Optimized Classical Solver (PuLP)] Initiated ---")
    
    # 1. Initialize PuLP Problem
    prob = pulp.LpProblem("CQM_to_PuLP", pulp.LpMinimize)
    
    # 2. Create PuLP variables based on the CQM
    pulp_vars = {}
    for var in cqm.variables:
        if cqm.vartype(var) != dimod.BINARY:
            raise ValueError(f"Variable {var} is not BINARY. This solver expects binary problems.")
        pulp_vars[var] = pulp.LpVariable(var, cat=pulp.LpBinary)
        
    # 3. Translate Objective (with safeguard for linear-only models)
    if cqm.objective.quadratic:
        raise ValueError("CQM contains quadratic objective terms! PuLP only solves linear problems. Aborting.")
        
    obj_expr = sum(bias * pulp_vars[var] for var, bias in cqm.objective.linear.items())
    obj_expr += cqm.objective.offset
    prob += obj_expr
    
    # 4. Translate Constraints (with safeguard for linear-only constraints)
    for label, constraint in cqm.constraints.items():
        if constraint.lhs.quadratic:
             raise ValueError(f"Constraint {label} contains quadratic terms! PuLP requires linear constraints.")
             
        lhs_expr = sum(bias * pulp_vars[var] for var, bias in constraint.lhs.linear.items())
        lhs_expr += constraint.lhs.offset # <--- FIX 2: Add the constraint offset!
        
        rhs = constraint.rhs
        sense = constraint.sense.value
        
        if sense == '<=': prob += (lhs_expr <= rhs, label)
        elif sense == '>=': prob += (lhs_expr >= rhs, label)
        elif sense == '==': prob += (lhs_expr == rhs, label)
            
    # 5. Solve and tell CBC to write a log file so we can parse the exact nodes explored
    log_file = "cbc_solve.log"
    solver = pulp.PULP_CBC_CMD(msg=True, logPath=log_file, presolve=False)
    
    start_time = time.perf_counter()
    prob.solve(solver)
    end_time = time.perf_counter()
    
    # 6. Parse the log for exact node count and iterations (Bulletproof Regex)
    nodes_explored = 0
    simplex_iterations = 0
    
    if os.path.exists(log_file):
        with open(log_file, "r") as f:
            log_text = f.read()
            
            # Hunt for any number associated with "iterations" 
            # Matches: "Total iterations: 5", "5 iterations", "iterations 5"
            iter_match = re.search(r'(?:iterations?.*?(\d+))|(?:(\d+).*?iterations?)', log_text, re.IGNORECASE)
            if iter_match:
                # Grab whichever regex group actually caught the number
                simplex_iterations = int(iter_match.group(1) or iter_match.group(2))
                
            # Hunt for any number associated with "nodes"
            # Matches: "Enumerated nodes: 2", "2 nodes", "nodes 2"
            node_match = re.search(r'(?:nodes?.*?(\d+))|(?:(\d+).*?nodes?)', log_text, re.IGNORECASE)
            if node_match:
                nodes_explored = int(node_match.group(1) or node_match.group(2))
                
        os.remove(log_file) # Clean up the temp file
    
    # 7. Extract Results and Unified Telemetry
    best_cost = pulp.value(prob.objective)
    
    if pulp.LpStatus[prob.status] == 'Optimal':
        validity = f"Exact Global Optimum Verified (Cost: {best_cost})"
        best_sol = {var: int(pulp_vars[var].varValue) for var in cqm.variables}
        best_sols = [best_sol]
    else:
        validity = f"Solver Status: {pulp.LpStatus[prob.status]}"
        best_sols, best_cost = [], None

    # New Paradigm-Specific Metrics Schema
    metrics = {
        "execution_time_seconds": end_time - start_time,
        "validity": validity,
        "paradigm": "Classical Branch-and-Bound",
        "paradigm_details": {
            "nodes_explored": nodes_explored,
            "simplex_iterations": simplex_iterations
        }
    }
    
    print(f"[Optimized Solver] Search complete.")
    print(f"  -> Paradigm:     {metrics['paradigm']}")
    print(f"  -> Nodes:        {metrics['paradigm_details']['nodes_explored']}")
    print(f"  -> Simplex Iter: {metrics['paradigm_details']['simplex_iterations']}")
    print(f"  -> Time:         {metrics['execution_time_seconds']:.6f} seconds")
    print(f"  -> Validity:     {metrics['validity']}\n")
    
    return best_cost, best_sols, metrics

# --- Execution Block ---
if __name__ == "__main__":
    print("--- RUNNING STANDALONE OPTIMIZED SOLVER TEST ---")
    
    print("[1] Creating a dummy 3-variable problem...")
    test_cqm = dimod.ConstrainedQuadraticModel()
    x = [dimod.Binary(f"x{i}") for i in range(3)]
    
    test_cqm.set_objective(x[0] - x[1] + 2*x[2])
    test_cqm.add_constraint(x[0] + x[1] == 1, label='c1')
    
    print("[2] Passing dummy model to the solver...\n")
    best_cost, best_solutions, metrics = solve_classical_optimized(test_cqm)
    
    print(">>> FINAL TEST RESULTS <<<")
    print(f"Minimum Cost: {best_cost}")
    print(f"Optimal Solutions Found: {len(best_solutions)}")
    for idx, sol in enumerate(best_solutions, 1):
        print(f"  Sol {idx}: {sol}")