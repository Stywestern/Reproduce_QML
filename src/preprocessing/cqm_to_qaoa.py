############################################################################################
#                                       Imports
############################################################################################

# Third Party Libraries
import dimod
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters.quadratic_program_to_qubo import QuadraticProgramToQubo

############################################################################################
#                                       Main Block
############################################################################################

def cqm_to_qaoa_operator(cqm: dimod.ConstrainedQuadraticModel, lagrange_mult: float = 3.0):
    """
    Translates dimod.CQM -> Qiskit QP -> Penalty QUBO -> Ising Hamiltonian.
    """
    qp = QuadraticProgram()
    original_vars = list(cqm.variables)

    # 1. Decision Variables
    for var in cqm.variables:
        if cqm.vartype(var) != dimod.BINARY:
            raise ValueError(f"Variable '{var}' is not BINARY.")
        qp.binary_var(var)

    # 2. Objective
    linear_obj = {var: bias for var, bias in cqm.objective.linear.items()}
    qp.minimize(constant=cqm.objective.offset, linear=linear_obj)

    # 3. Constraints
    for label, constraint in cqm.constraints.items():
        lhs_linear = {var: bias for var, bias in constraint.lhs.linear.items()}
        offset = constraint.lhs.offset if hasattr(constraint.lhs, 'offset') else 0
        rhs = constraint.rhs - offset

        sense_map = {'<=': '<=', '>=': '>=', '==': '=='}
        sense = sense_map.get(constraint.sense.value)

        qp.linear_constraint(linear=lhs_linear, sense=sense, rhs=rhs, name=label)

    # Convert to QUBO
    qubo_converter = QuadraticProgramToQubo(penalty=lagrange_mult)
    qubo_qp = qubo_converter.convert(qp)

    # Convert QUBO to Ising Hamiltonian
    op, offset = qubo_qp.to_ising()

    return op, offset, qubo_qp, original_vars