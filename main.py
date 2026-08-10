############################################################################################
#                                   MAIN BENCHMARK RUNNER
############################################################################################

from src.problems.math_baseline import AppendixAModel

# Transformers
from src.preprocessing.cqm_to_pulp import cqm_to_pulp
from src.preprocessing.cqm_to_qaoa import cqm_to_qaoa_operator
from src.preprocessing.cqm_to_bqm import cqm_to_bqm_slack

# Algorithms
from src.algorithms.brute_force import run_brute_force
from src.algorithms.pulp_branch_bound import run_pulp_branch_bound
from src.algorithms.qaoa_gate_sim import run_qaoa_gate_sim
from src.algorithms.qa_neal_sim import run_neal_simulated_annealing
from src.algorithms.qphr_alm_sim import run_qphr_alm_loop

# Benchmarking
from src.benchmarks.metrics_collector import MetricsCollector

def main():
    print("==========================================================================================")
    print("                     STARTING QUANTUM POWER SYSTEMS BENCHMARK SUITE                       ")
    print("==========================================================================================\n")

    # 1. Load Problem Instance (Pillar 1)
    problem = AppendixAModel()
    cqm = problem.get_cqm()

    all_metrics = []

    # ----------------------------------------------------------------------------------------
    # [Run 1]: Classical Brute Force
    # ----------------------------------------------------------------------------------------
    cost1, sols1, time1, details1 = run_brute_force(cqm)
    m1 = MetricsCollector.format_results("Classical Brute Force", cost1, sols1, time1, details1)
    all_metrics.append(m1)

    # ----------------------------------------------------------------------------------------
    # [Run 2]: Classical Branch-and-Bound (PuLP/CBC)
    # ----------------------------------------------------------------------------------------
    pulp_prob, pulp_vars = cqm_to_pulp(cqm)
    cost2, sols2, time2, details2 = run_pulp_branch_bound(pulp_prob, pulp_vars)
    m2 = MetricsCollector.format_results("Classical Branch-and-Bound (PuLP)", cost2, sols2, time2, details2)
    all_metrics.append(m2)

    # ----------------------------------------------------------------------------------------
    # [Run 3]: Quantum Annealing with Standard Slacks (Neal)
    # ----------------------------------------------------------------------------------------
    bqm, invert_map, orig_vars, num_slacks = cqm_to_bqm_slack(cqm, lagrange_mult=3.0)
    cost3, sols3, time3, details3 = run_neal_simulated_annealing(
        bqm=bqm, cqm=cqm, original_vars=orig_vars, num_slacks=num_slacks, num_reads=1000, num_sweeps=1000, seed=2
    )
    m3 = MetricsCollector.format_results("Quantum Annealing (Standard Slacks)", cost3, sols3, time3, details3)
    all_metrics.append(m3)

    # ----------------------------------------------------------------------------------------
    # [Run 4]: Quantum Gate-Based (QAOA Statevector)
    # ----------------------------------------------------------------------------------------
    op, offset, qubo_qp, orig_vars_qaoa = cqm_to_qaoa_operator(cqm, lagrange_mult=3.0)
    cost4, sols4, time4, details4 = run_qaoa_gate_sim(
        op=op, offset=offset, qubo_qp=qubo_qp, cqm=cqm, original_vars=orig_vars_qaoa, reps=2, seed=2
    )
    m4 = MetricsCollector.format_results("Quantum Gate-Based (QAOA)", cost4, sols4, time4, details4)
    all_metrics.append(m4)

    # ----------------------------------------------------------------------------------------
    # [Run 5]: Paper Innovation (QPHR-ALM Adaptive Hybrid)
    # ----------------------------------------------------------------------------------------
    cost5, sols5, time5, details5 = run_qphr_alm_loop(
        cqm=cqm, sigma_0=2.0, eta=1.5, rho=0.8, delta=1e-3, max_iters=20, seed=2
    )
    m5 = MetricsCollector.format_results("Quantum Hybrid Benders (QPHR-ALM)", cost5, sols5, time5, details5)
    all_metrics.append(m5)

    # ----------------------------------------------------------------------------------------
    # Final Showdown Table
    # ----------------------------------------------------------------------------------------
    MetricsCollector.print_benchmark_table(all_metrics)


if __name__ == "__main__":
    main()