############################################################################################
#                                       Imports
############################################################################################

# Native Libraries
import time
import numpy as np

# Modules
from src.tools.qphr_math import (
    parse_cqm_constraints,
    evaluate_g_i,
    build_qphr_bqm,
    compute_residual_R
)

# Third Party Libraries
from dwave.system import DWaveSampler, EmbeddingComposite, FixedEmbeddingComposite
import dimod
import neal

############################################################################################
#                                       Main Block
############################################################################################

def run_qphr_alm_loop(
    cqm: dimod.ConstrainedQuadraticModel,
    sigma_0: float = 2.0,
    eta: float = 1.5,
    rho: float = 0.8,
    delta: float = 1e-3,
    max_iters: int = 20,
    num_reads: int = 1000,
    num_sweeps: int = 1000,
    seed: int = 2
):
    """
    Executes Algorithm 1 (QPHR-ALM) Powell-Hestenes-Rockafellar multiplier loop.

    Returns:
        best_cost (float): Unpenalized optimal cost value.
        best_sols (list[dict]): List containing the optimal decision variable dictionary.
        execution_time (float): Total loop execution time in seconds.
        raw_details (dict): Iteration history, residuals, and slack-free qubit counts.
    """
    parsed_constraints = parse_cqm_constraints(cqm)
    num_constraints = len(parsed_constraints)
    original_vars = list(cqm.variables)
    num_qubits = len(original_vars)

    lambda_vec = np.zeros(num_constraints)
    sigma = sigma_0
    l = 1
    last_sample = None
    R_prev = float('inf')

    sampler = neal.SimulatedAnnealingSampler()
    iteration_history = []

    start_time = time.perf_counter()

    while l <= max_iters:
        # 1. Dynamically build slack-free BQM
        bqm = build_qphr_bqm(cqm, parsed_constraints, lambda_vec, sigma, last_sample)

        # 2. Sample on Quantum / Classical Annealer
        sampleset = sampler.sample(
            bqm,
            num_reads=num_reads,
            num_sweeps=num_sweeps,
            seed=seed + l
        )

        best_sample_record = sampleset.first.sample
        xl_sample = {var: int(best_sample_record[var]) for var in original_vars if var in best_sample_record}

        # 3. Calculate Residual & Feasibility
        R_l = compute_residual_R(parsed_constraints, xl_sample, lambda_vec, sigma)
        current_cost = float(cqm.objective.energy(xl_sample))
        is_feasible = cqm.check_feasible(xl_sample)

        iteration_history.append({
            'iter': l,
            'cost': current_cost,
            'residual': R_l,
            'sigma': sigma,
            'feasible': is_feasible
        })

        # 4. Convergence Termination Check
        if R_l <= delta and is_feasible:
            last_sample = xl_sample
            break

        # 5. Multiplier & Penalty Parameter Updates
        for i, constr in enumerate(parsed_constraints):
            g_val = evaluate_g_i(constr, xl_sample)
            lambda_vec[i] = max(0.0, lambda_vec[i] + sigma * g_val)

        if l > 1 and R_l >= rho * R_prev:
            sigma = eta * sigma

        R_prev = R_l
        last_sample = xl_sample
        l += 1

    end_time = time.perf_counter()
    execution_time = end_time - start_time

    best_sol = last_sample if last_sample is not None else xl_sample
    best_cost = float(cqm.objective.energy(best_sol))
    best_sols = [best_sol] if best_sol and cqm.check_feasible(best_sol) else []

    raw_details = {
        "logical_qubits": num_qubits,
        "total_qphr_iterations": len(iteration_history),
        "final_residual_R": round(R_l, 6),
        "final_sigma": round(sigma, 2),
        "seed": seed
    }

    return best_cost, best_sols, execution_time, raw_details


############################################################################################
#                                  Execution Block
############################################################################################
if __name__ == "__main__":
    from src.problems.math_baseline import AppendixAModel
    from src.benchmarks.metrics_collector import MetricsCollector

    # 1. Load Problem
    problem = AppendixAModel()
    cqm = problem.get_cqm()

    # 2. Run QPHR-ALM Loop Algorithm
    cost, sols, exec_time, details = run_qphr_alm_loop(
        cqm=cqm,
        sigma_0=2.0,
        eta=1.5,
        rho=0.8,
        delta=1e-3,
        max_iters=20,
        seed=2
    )

    # 3. Format Metrics via MetricsCollector
    metrics = MetricsCollector.format_results(
        paradigm_name="Quantum Hybrid Benders (QPHR-ALM Paper Algorithm)",
        best_cost=cost,
        best_sols=sols,
        execution_time=exec_time,
        raw_details=details
    )