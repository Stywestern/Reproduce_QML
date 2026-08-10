############################################################################################
#                       ALGORITHMS: GATE-BASED QAOA SIMULATOR
############################################################################################

import time
import warnings
import numpy as np

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from qiskit_algorithms import SamplingVQE
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import StatevectorSampler

from src.tools.qaoa_ansatz_builder import build_qaoa_ansatz


def run_qaoa_gate_sim(op, offset: float, qubo_qp, cqm, original_vars, reps: int = 2, seed: int = 2):
    """
    Executes QAOA statevector simulation on a Qiskit Ising operator.
    """
    ansatz_circuit = build_qaoa_ansatz(op=op, reps=reps)
    decomposed_circuit = ansatz_circuit.decompose(reps=reps)

    exact_qubits = decomposed_circuit.num_qubits
    exact_depth = decomposed_circuit.depth()
    gate_counts = dict(decomposed_circuit.count_ops())

    np.random.seed(seed)
    initial_point = np.random.uniform(0, np.pi, size=2 * reps)

    optimizer = COBYLA(maxiter=100, rhobeg=1.0, tol=1e-4)
    sampler = StatevectorSampler(default_shots=1024, seed=seed)

    vqe = SamplingVQE(sampler=sampler, ansatz=ansatz_circuit, optimizer=optimizer, initial_point=initial_point)

    start_time = time.perf_counter()
    result = vqe.compute_minimum_eigenvalue(op)
    end_time = time.perf_counter()
    execution_time = end_time - start_time

    # 1. Extract Measurement Bitstring
    best_measurement = getattr(result, 'best_measurement', {})
    raw_bitstring = best_measurement.get('bitstring', '')
    binary_values = [int(b) for b in reversed(raw_bitstring)] if raw_bitstring else []

    # 2. Filter ONLY Original Decision Variables (x1..x6), ignore slack qubits
    best_sol = {}
    for i, var_obj in enumerate(qubo_qp.variables):
        if var_obj.name in original_vars and i < len(binary_values):
            best_sol[var_obj.name] = binary_values[i]

    # 3. Calculate True Unpenalized Original Objective Cost f(x)
    if best_sol and len(best_sol) == len(original_vars):
        best_cost = float(cqm.objective.energy(best_sol))
        best_sols = [best_sol]
    else:
        best_cost = float('nan')
        best_sols = []

    evaluations_used = getattr(result, 'optimizer_evals', getattr(result, 'cost_function_evals', 'N/A'))
    
    if hasattr(result, 'optimal_point') and result.optimal_point is not None:
        best_params = [round(float(angle), 4) for angle in result.optimal_point]
    else:
        best_params = "N/A"

    raw_details = {
        "qubits_required": exact_qubits,
        "circuit_depth": exact_depth,
        "optimizer_evaluations": evaluations_used,
        "best_parameters": str(best_params),
        "gate_counts": str(gate_counts),
        "seed": seed
    }

    return best_cost, best_sols, execution_time, raw_details


############################################################################################
#                               Execution / Verification Block
############################################################################################
if __name__ == "__main__":
    from src.problems.math_baseline import AppendixAModel
    from src.preprocessing.cqm_to_qaoa import cqm_to_qaoa_operator
    from src.benchmarks.metrics_collector import MetricsCollector

    # 1. Load Problem
    problem = AppendixAModel()
    cqm = problem.get_cqm()

    # 2. Preprocess / Transform Problem
    op, offset, qubo_qp, original_vars = cqm_to_qaoa_operator(cqm, lagrange_mult=3.0)

    # 3. Run Pure Algorithm
    cost, sols, exec_time, details = run_qaoa_gate_sim(
        op=op, 
        offset=offset, 
        qubo_qp=qubo_qp, 
        cqm=cqm, 
        original_vars=original_vars, 
        reps=2, 
        seed=2
    )

    # 4. Format Metrics via MetricsCollector
    metrics = MetricsCollector.format_results(
        paradigm_name="Quantum Gate-Based (QAOA Statevector)",
        best_cost=cost,
        best_sols=sols,
        execution_time=exec_time,
        raw_details=details
    )