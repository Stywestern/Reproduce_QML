############################################################################################
#                                           IMPORTS 
############################################################################################

from problems.math_baseline import build_appendix_a_model
from solvers.brute_force import solve_brute_force
from solvers.classical_optimized import solve_classical_optimized

############################################################################################
#                                       Appendix Functions
############################################################################################

def format_solutions(sol_list):
    """Formats a list of solution dictionaries into a comma-separated list of binary strings."""
    if not sol_list:
        return "None"
    formatted = []
    for sol_dict in sol_list:
        # Sort keys to ensure x1 comes first, x6 comes last
        sorted_keys = sorted(sol_dict.keys(), key=lambda k: int(k[1:]) if k[1:].isdigit() else k)
        binary_str = "".join(str(sol_dict[k]) for k in sorted_keys)
        formatted.append(binary_str)
    return ", ".join(formatted)

############################################################################################
#                                       Execution Block
############################################################################################
if __name__ == "__main__":
    print("==================================================")
    print("  QML SUC Project: Solver Showdown (Appendix A)   ")
    print("==================================================")
    
    # 1. Instantiate the problem (The "What")
    print("\n[PHASE 1: PROBLEM DEFINITION]")
    model = build_appendix_a_model()
    
    # 2. Pass it to the Brute Force Solver
    print("\n[PHASE 2: BRUTE FORCE SOLVER]")
    bf_cost, bf_sols, bf_metrics = solve_brute_force(model)
    
    # 3. Pass it to the Optimized Solver
    print("\n[PHASE 3: OPTIMIZED SOLVER (PuLP)]")
    opt_cost, opt_sols, opt_metrics = solve_classical_optimized(model)
    
    # 4. Output Benchmarking Table
    print("\n==========================================================================================")
    print("                                 BENCHMARKING RESULTS")
    print("==========================================================================================")
    
    # Table Header
    print(f"{'Metric':<25} | {'Brute Force':<30} | {'Optimized (Branch & Bound)':<30}")
    print("-" * 90)
    
    # Row 1: Cost
    print(f"{'Best Cost Found':<25} | {str(bf_cost):<30} | {str(opt_cost):<30}")
    
    # Row 2: Execution Time
    bf_time = f"{bf_metrics['execution_time_seconds']:.6f} s"
    opt_time = f"{opt_metrics['execution_time_seconds']:.6f} s"
    print(f"{'Execution Time':<25} | {bf_time:<30} | {opt_time:<30}")
    
    # Row 3: Solutions Found (Replaced Validity)
    bf_str_sols = format_solutions(bf_sols)
    opt_str_sols = format_solutions(opt_sols)
    print(f"{'Solutions Found':<25} | {bf_str_sols:<30} | {opt_str_sols:<30}")
    
    # Row 4: Paradigm-Specific Work
    bf_work = f"{bf_metrics['paradigm_details'].get('combinations_evaluated', 'N/A')} combinations"
    
    nodes = opt_metrics['paradigm_details'].get('nodes_explored', 'N/A')
    iters = opt_metrics['paradigm_details'].get('simplex_iterations', 'N/A')
    opt_work = f"{nodes} nodes, {iters} simplex iters"
    
    print(f"{'Algorithmic Work':<25} | {bf_work:<30} | {opt_work:<30}")
    print("==========================================================================================\n")