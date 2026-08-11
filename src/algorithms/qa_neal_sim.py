############################################################################################
#                                       Imports
############################################################################################

# Native Libraries
import time
import warnings

# Third Party Libraries
import dimod
import neal

# Setup
warnings.filterwarnings("ignore")

############################################################################################
#                                       Main Block
############################################################################################

def run_neal_simulated_annealing(
    bqm: dimod.BinaryQuadraticModel,
    cqm: dimod.ConstrainedQuadraticModel,
    original_vars: list,
    num_slacks: int,
    num_reads: int = 1000,
    num_sweeps: int = 1000,
    seed: int = 2
):
    """
    Executes D-Wave's Simulated Annealer (neal) over a target BinaryQuadraticModel.

    Args:
        bqm (dimod.BinaryQuadraticModel): Target BQM Hamiltonian.
        cqm (dimod.ConstrainedQuadraticModel): Original CQM model for feasibility checking.
        original_vars (list): Original decision variable keys.
        num_slacks (int): Count of slack variables.
        num_reads (int): Total annealing cycles.
        num_sweeps (int): Monte Carlo sweeps per read.
        seed (int): Seed for reproducibility.

    Returns:
        best_cost (float): Unpenalized objective cost of the best feasible state.
        best_sols (list[dict]): List containing the optimal variable assignment dictionary.
        execution_time (float): Sampling runtime in seconds.
        raw_details (dict): Topology metrics, yield frequencies, and execution parameters.
    """

    logical_qubits = len(bqm.variables)
    sampler = neal.SimulatedAnnealingSampler()

    start_time = time.perf_counter()
    sampleset = sampler.sample(
        bqm,
        num_reads=num_reads,
        num_sweeps=num_sweeps,
        beta_schedule_type="geometric",
        seed=seed
    )
    end_time = time.perf_counter()
    execution_time = end_time - start_time

    # --- 1. Topology & Graph Metrics ---
    num_linear_bias = len(bqm.linear)
    num_quadratic_couplers = len(bqm.quadratic)
    graph_density = (
        num_quadratic_couplers / (logical_qubits * (logical_qubits - 1) / 2)
        if logical_qubits > 1 else 0.0
    )

    # --- 2. Pass 1: Identify Best Feasible Candidate State ---
    best_sol = None
    best_cost = float('inf')
    best_bqm_energy = float('inf')
    feasible_read_count = 0
    total_reads = sum(sampleset.record.num_occurrences)

    sample_data = list(sampleset.data(['sample', 'energy', 'num_occurrences']))

    for sample, energy, num_occurrences in sample_data:
        candidate_sol = {var: int(sample[var]) for var in original_vars if var in sample}

        if cqm.check_feasible(candidate_sol):
            feasible_read_count += num_occurrences
            cost = float(cqm.objective.energy(candidate_sol))

            if cost < best_cost:
                best_cost = cost
                best_sol = candidate_sol
                best_bqm_energy = float(energy)

    # --- 3. Pass 2: Accumulate Exact Ground State Frequencies ---
    ground_state_occurrences = 0
    if best_sol is not None:
        for sample, energy, num_occurrences in sample_data:
            candidate_sol = {var: int(sample[var]) for var in original_vars if var in sample}
            if cqm.check_feasible(candidate_sol):
                cost = float(cqm.objective.energy(candidate_sol))
                if abs(cost - best_cost) < 1e-6:
                    ground_state_occurrences += num_occurrences

        best_sols = [best_sol]
    else:
        best_cost = float('nan')
        best_sols = []

    ground_state_prob = (ground_state_occurrences / total_reads) * 100 if total_reads > 0 else 0.0
    feasibility_yield = (feasible_read_count / total_reads) * 100 if total_reads > 0 else 0.0

    raw_details = {
        "logical_qubits": logical_qubits,
        "slack_variables": num_slacks,
        "quadratic_couplers": num_quadratic_couplers,
        "graph_density": f"{graph_density * 100:.2f}%",
        "ground_state_frequency": f"{ground_state_occurrences}/{total_reads} ({ground_state_prob:.2f}%)",
        "feasibility_yield": f"{feasible_read_count}/{total_reads} ({feasibility_yield:.2f}%)",
        "raw_bqm_energy": round(best_bqm_energy, 4) if best_sol else "N/A",
        "num_reads": num_reads,
        "num_sweeps": num_sweeps,
        "seed": seed
    }

    return best_cost, best_sols, execution_time, raw_details


############################################################################################
#                                  Execution Block
############################################################################################
if __name__ == "__main__":
    from src.problems.math_baseline import AppendixAModel
    from src.preprocessing.cqm_to_bqm import cqm_to_bqm_slack
    from src.benchmarks.metrics_collector import MetricsCollector

    # 1. Load Problem
    problem = AppendixAModel()
    cqm = problem.get_cqm()

    # 2. Preprocess / Transform Problem
    bqm, invert_map, original_vars, num_slacks = cqm_to_bqm_slack(cqm, lagrange_mult=3.0)

    # 3. Run Pure Algorithm
    cost, sols, exec_time, details = run_neal_simulated_annealing(
        bqm=bqm,
        cqm=cqm,
        original_vars=original_vars,
        num_slacks=num_slacks,
        num_reads=1000,
        num_sweeps=1000,
        seed=2
    )

    # 4. Format Metrics in Priority Order via MetricsCollector
    metrics = MetricsCollector.format_results(
        paradigm_name="Quantum Annealing (Simulated - Neal)",
        best_cost=cost,
        best_sols=sols,
        execution_time=exec_time,
        raw_details=details
    )