############################################################################################
#                                          Imports
############################################################################################

import os
import re
import time
import pulp

############################################################################################
#                                          Main Block
############################################################################################

def run_pulp_branch_bound(prob: pulp.LpProblem, pulp_vars: dict, log_file: str = "cbc_solve.log"):
    """
    Executes the PuLP CBC Branch-and-Bound algorithm on a pre-constructed pulp.LpProblem.
    
    Args:
        prob (pulp.LpProblem): Pure PuLP problem object.
        pulp_vars (dict): Variable dictionary mapping variable names to pulp.LpVariable.
        log_file (str): Temporary log file path to extract solver telemetry.

    Returns:
        best_cost (float): Optimal cost value.
        best_sols (list[dict]): List containing the optimal variable assignment dictionary.
        execution_time (float): Solver execution time in seconds.
        raw_details (dict): Raw execution details for benchmarking.
    """

    solver = pulp.PULP_CBC_CMD(msg=False, logPath=log_file, presolve=False)

    start_time = time.perf_counter()
    prob.solve(solver)
    end_time = time.perf_counter()
    execution_time = end_time - start_time

    # Parse CBC Log File for Pure Branch-and-Bound Metrics
    nodes_explored = 0
    simplex_iterations = 0

    if os.path.exists(log_file):
        with open(log_file, "r") as f:
            log_text = f.read()

            iter_match = re.search(r'(?:iterations?.*?(\d+))|(?:(\d+).*?iterations?)', log_text, re.IGNORECASE)
            if iter_match:
                simplex_iterations = int(iter_match.group(1) or iter_match.group(2))

            node_match = re.search(r'(?:nodes?.*?(\d+))|(?:(\d+).*?nodes?)', log_text, re.IGNORECASE)
            if node_match:
                nodes_explored = int(node_match.group(1) or node_match.group(2))

        try:
            os.remove(log_file)
        except OSError:
            pass

    # Extract raw solution state
    status_str = pulp.LpStatus[prob.status]
    if status_str == 'Optimal':
        best_cost = float(pulp.value(prob.objective))
        best_sol = {var_name: int(var_obj.varValue) for var_name, var_obj in pulp_vars.items()}
        best_sols = [best_sol]
    else:
        best_cost = float('nan')
        best_sols = []

    raw_details = {
        "nodes_explored": nodes_explored,
        "simplex_iterations": simplex_iterations,
        "pulp_status": status_str
    }

    return best_cost, best_sols, execution_time, raw_details


############################################################################################
#                               Execution / Verification Block
############################################################################################
if __name__ == "__main__":
    from src.problems.math_baseline import AppendixAModel
    from src.preprocessing.cqm_to_pulp import cqm_to_pulp
    from src.benchmarks.metrics_collector import MetricsCollector

    # 1. Load Problem
    problem = AppendixAModel()
    cqm = problem.get_cqm()

    # 2. Preprocess / Transform Problem
    pulp_prob, pulp_vars = cqm_to_pulp(cqm)

    # 3. Run Pure Algorithm 
    cost, sols, exec_time, details = run_pulp_branch_bound(pulp_prob, pulp_vars)

    # 4. Format Metrics in Priority Order via MetricsCollector
    metrics = MetricsCollector.format_results(
        paradigm_name="Classical Branch-and-Bound (PuLP/CBC)",
        best_cost=cost,
        best_sols=sols,
        execution_time=exec_time,
        raw_details=details
    )