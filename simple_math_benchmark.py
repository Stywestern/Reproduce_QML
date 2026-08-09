import itertools
import pulp
import dimod
import neal
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_algorithms import QAOA
from qiskit.primitives import Sampler

def solve_brute_force():
    print("--- 1. Brute Force ---")
    best_cost = float('inf')
    best_sols = []
    
    # Iterate through all 2^6 = 64 possible binary combinations
    for x in itertools.product([0, 1], repeat=6):
        # Apply the constraints
        c1 = -2*x[1] - 2*x[4] - x[5] + 3 <= 0
        c2 = -x[0] + x[2] - x[3] + 2*x[5] <= 0
        c3 = -x[0] + x[2] + x[3] <= 0
        
        if c1 and c2 and c3:
            # Calculate objective function
            cost = 6*x[0] + 3*x[1] - 5*x[2] - 6*x[3] + 4*x[4] - 7*x[5]
            if cost < best_cost:
                best_cost = cost
                best_sols = [x]
            elif cost == best_cost:
                best_sols.append(x)
                
    for sol in best_sols:
        print(f"Solution: {sol}, Cost: {best_cost}")
    print()

def solve_classical_optimized():
    print("--- 2. Classical Optimized (PuLP) ---")
    # Create the linear programming problem
    prob = pulp.LpProblem("Baseline_Appendix_A", pulp.LpMinimize)
    
    # Create 6 binary variables
    x = [pulp.LpVariable(f"x{i+1}", cat='Binary') for i in range(6)]
    
    # Objective function
    prob += 6*x[0] + 3*x[1] - 5*x[2] - 6*x[3] + 4*x[4] - 7*x[5]
    
    # Constraints
    prob += -2*x[1] - 2*x[4] - x[5] + 3 <= 0
    prob += -x[0] + x[2] - x[3] + 2*x[5] <= 0
    prob += -x[0] + x[2] + x[3] <= 0
    
    # Solve
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    
    sol = tuple(int(v.varValue) for v in prob.variables())
    print(f"Solution: {sol}, Cost: {pulp.value(prob.objective)}")
    print()

def solve_quantum_gate_qaoa():
    print("--- 3. Quantum Gate-Based Simulation (QAOA) ---")
    # Formulate as a Quadratic Program for Qiskit
    qp = QuadraticProgram()
    for i in range(1, 7):
        qp.binary_var(f"x{i}")
        
    qp.minimize(linear={"x1": 6, "x2": 3, "x3": -5, "x4": -6, "x5": 4, "x6": -7})
    
    qp.linear_constraint(linear={"x2": -2, "x5": -2, "x6": -1}, sense="<=", rhs=-3)
    qp.linear_constraint(linear={"x1": -1, "x3": 1, "x4": -1, "x6": 2}, sense="<=", rhs=0)
    qp.linear_constraint(linear={"x1": -1, "x3": 1, "x4": 1}, sense="<=", rhs=0)
    
    # Setup QAOA with a classical simulator backend
    qaoa = QAOA(sampler=Sampler(), reps=2)
    optimizer = MinimumEigenOptimizer(qaoa)
    
    # Solve
    result = optimizer.solve(qp)
    sol = tuple(int(val) for val in result.x)
    print(f"Solution: {sol}, Cost: {result.fval}")
    print()

def solve_quantum_annealing_sim():
    print("--- 4. Quantum Annealing Simulation (Simulated Annealing) ---")
    # Define a Constrained Quadratic Model (CQM) for D-Wave
    cqm = dimod.ConstrainedQuadraticModel()
    
    x = [dimod.Binary(f"x{i+1}") for i in range(6)]
    
    cqm.set_objective(6*x[0] + 3*x[1] - 5*x[2] - 6*x[3] + 4*x[4] - 7*x[5])
    
    cqm.add_constraint(-2*x[1] - 2*x[4] - x[5] + 3 <= 0, label='c1')
    cqm.add_constraint(-x[0] + x[2] - x[3] + 2*x[5] <= 0, label='c2')
    cqm.add_constraint(-x[0] + x[2] + x[3] <= 0, label='c3')
    
    # Convert constraints to penalties to make it a QUBO (Binary Quadratic Model)
    # The penalty method requires slack variables for inequalities, which is exactly
    # what the paper's QPHR-ALM method tries to avoid! But for this simulation, 
    # we use the standard D-Wave converter.
    bqm, invert = dimod.cqm_to_bqm(cqm)
    
    # Solve using D-Wave's local Simulated Annealing sampler
    sampler = neal.SimulatedAnnealingSampler()
    sampleset = sampler.sample(bqm, num_reads=1000)
    
    # Filter for feasible solutions
    feasible_samples = []
    for sample, energy in sampleset.data(['sample', 'energy']):
        # Recover original variables from the BQM (removing the added slack variables)
        recovered_sample = invert(sample)
        if cqm.check_feasible(recovered_sample):
            feasible_samples.append((recovered_sample, cqm.objective.energy(recovered_sample)))
            
    # Sort by cost and grab the best one
    feasible_samples.sort(key=lambda item: item[1])
    best_sample, best_cost = feasible_samples[0]
    
    sol = tuple(best_sample[f"x{i}"] for i in range(1, 7))
    print(f"Solution: {sol}, Cost: {best_cost}")

if __name__ == "__main__":
    solve_brute_force()
    solve_classical_optimized()
    solve_quantum_gate_qaoa()
    solve_quantum_annealing_sim()