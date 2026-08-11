############################################################################################
#                                       Imports
############################################################################################

# Third Party Libraries
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import QAOAAnsatz

############################################################################################
#                                       Main Block
############################################################################################


def build_qaoa_ansatz(
    op, 
    reps: int = 1, 
    ansatz_type: str = "manual_standard", 
    mixer_type: str = "x_mixer"
) -> QuantumCircuit:
    """
    Constructs a QAOA ansatz circuit with configurable expressivity options.

    Args:
        op: Qiskit SparsePauliOp cost Hamiltonian operator.
        reps (int): Number of QAOA p-layers.
        ansatz_type (str): 
            - 'manual_standard' : Explicit gate-by-gate decomposition.
            - 'qiskit_library'  : High-level Qiskit QAOAAnsatz synthesis.
            - 'multi_angle'     : Multi-angle QAOA (ma-QAOA) with per-gate parameters.
        mixer_type (str): Mixer unitary type ('x_mixer').

    Returns:
        QuantumCircuit: Parametrized QAOA circuit.
    """
    num_qubits = op.num_qubits

    # Option 1: Native High-Level Qiskit Library Ansatz
    if ansatz_type == "qiskit_library":
        return QAOAAnsatz(cost_operator=op, reps=reps)

    qc = QuantumCircuit(num_qubits)

    # Option 2 & 3: Custom Explicit Gate Decompositions
    if ansatz_type == "manual_standard":
        gammas = ParameterVector('γ', reps)
        betas = ParameterVector('β', reps)

        # Layer 0: Equal Superposition
        for q in range(num_qubits):
            qc.h(q)
        qc.barrier()

        # QAOA p-layers
        for p in range(reps):
            gamma = gammas[p]
            beta = betas[p]

            # --- COST UNITARY U(C, γ) ---
            for pauli_string, coeff in zip(op.paulis, op.coeffs):
                real_coeff = float(np.real(coeff))
                z_indices = [i for i, pauli in enumerate(reversed(str(pauli_string))) if pauli == 'Z']

                if len(z_indices) == 1:
                    qc.rz(2 * gamma * real_coeff, z_indices[0])
                elif len(z_indices) == 2:
                    q0, q1 = z_indices[0], z_indices[1]
                    qc.cx(q0, q1)
                    qc.rz(2 * gamma * real_coeff, q1)
                    qc.cx(q0, q1)

            qc.barrier()

            # --- MIXER UNITARY U(B, β) ---
            if mixer_type == "x_mixer":
                for q in range(num_qubits):
                    qc.rx(2 * beta, q)

            qc.barrier()

    elif ansatz_type == "multi_angle":
        # Multi-Angle QAOA: Each term gets its own parameter in each layer
        num_terms = len(op.paulis)
        gammas = ParameterVector('γ', reps * num_terms)
        betas = ParameterVector('β', reps * num_qubits)

        for q in range(num_qubits):
            qc.h(q)
        qc.barrier()

        term_idx = 0
        qubit_param_idx = 0

        for p in range(reps):
            # Cost Layer
            for pauli_string, coeff in zip(op.paulis, op.coeffs):
                gamma = gammas[term_idx]
                term_idx += 1
                real_coeff = float(np.real(coeff))
                z_indices = [i for i, pauli in enumerate(reversed(str(pauli_string))) if pauli == 'Z']

                if len(z_indices) == 1:
                    qc.rz(2 * gamma * real_coeff, z_indices[0])
                elif len(z_indices) == 2:
                    q0, q1 = z_indices[0], z_indices[1]
                    qc.cx(q0, q1)
                    qc.rz(2 * gamma * real_coeff, q1)
                    qc.cx(q0, q1)

            qc.barrier()

            # Mixer Layer
            for q in range(num_qubits):
                beta = betas[qubit_param_idx]
                qubit_param_idx += 1
                qc.rx(2 * beta, q)

            qc.barrier()

    return qc