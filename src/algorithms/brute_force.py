############################################################################################
#                                       Imports
############################################################################################

# Native Libraries
import itertools
import time

# Third Party Libraries
import dimod

############################################################################################
#                                       Main Block
############################################################################################

def run_brute_force(cqm: dimod.ConstrainedQuadraticModel):
    """
    Exact Classical Brute-Force Algorithm.
    Evaluates all 2^N binary combinations for a given ConstrainedQuadraticModel (CQM).
    
    Returns:
        best_cost (float): Lowest energy found.
        best_sols (list[dict]): List of optimal decision variable samples.
        execution_time (float): Time taken in seconds.
        raw_details (dict): Raw execution counts for benchmarking.
    """
    variables = list(cqm.variables)
    num_vars = len(variables)

    for var in variables:
        if cqm.vartype(var) != dimod.BINARY:
            raise ValueError(f"Variable '{var}' is not BINARY. Brute force aborted.")

    best_cost = float('inf')
    best_sols = []
    combinations_evaluated = 0

    start_time = time.perf_counter()

    # Pure Search Loop
    for vals in itertools.product([0, 1], repeat=num_vars):
        combinations_evaluated += 1
        sample = dict(zip(variables, vals))

        if cqm.check_feasible(sample):
            cost = float(cqm.objective.energy(sample))

            if cost < best_cost:
                best_cost = cost
                best_sols = [sample]
            elif cost == best_cost:
                best_sols.append(sample)

    end_time = time.perf_counter()
    execution_time = end_time - start_time

    # Raw telemetry details (pure data, no formatting or strings)
    raw_details = {
        "num_variables": num_vars,
        "combinations_evaluated": combinations_evaluated,
        "solutions_found_count": len(best_sols)
    }

    return best_cost, best_sols, execution_time, raw_details


############################################################################################
#                               Execution / Verification Block
############################################################################################
if __name__ == "__main__":
    from src.problems.math_baseline import AppendixAModel
    from src.benchmarks.metrics_collector import MetricsCollector

    # 1. Instantiate and build problem
    problem = AppendixAModel()
    cqm = problem.get_cqm()

    # 2. Run pure algorithm
    cost, sols, exec_time, details = run_brute_force(cqm)

    # 3. Process metrics and print telemetry via MetricsCollector
    metrics = MetricsCollector.format_results(
        paradigm_name="Classical Brute Force",
        best_cost=cost,
        best_sols=sols,
        execution_time=exec_time,
        raw_details=details
    )