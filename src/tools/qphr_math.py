############################################################################################
#                      UTILS: QPHR-ALM MATHEMATICAL CALCULATIONS
############################################################################################

import math
import numpy as np
import dimod


def parse_cqm_constraints(cqm: dimod.ConstrainedQuadraticModel) -> list[dict]:
    """
    Extracts linear inequality constraints from CQM into standard g_i(x) <= 0 form.
    g_i(x) = sum(a_j * x_j) + b <= 0
    """
    constraints_data = []

    for c_name, constraint in cqm.constraints.items():
        lhs = constraint.lhs
        rhs = constraint.rhs
        sense = constraint.sense.value

        linear_coeffs = {var: float(coeff) for var, coeff in lhs.linear.items()}
        constant_b = float(lhs.offset) - float(rhs)

        if sense == '>=':
            linear_coeffs = {v: -c for v, c in linear_coeffs.items()}
            constant_b = -constant_b

        constraints_data.append({
            'name': c_name,
            'a': linear_coeffs,
            'b': constant_b
        })

    return constraints_data


def evaluate_g_i(constraint: dict, sample: dict) -> float:
    """Calculates g_i(x) = sum(a_j * x_j) + b for a given binary sample."""
    val = constraint['b']
    for var, coeff in constraint['a'].items():
        val += coeff * int(sample.get(var, 0))
    return val


def build_qphr_bqm(
    cqm: dimod.ConstrainedQuadraticModel, 
    parsed_constraints: list, 
    lambda_vec: np.ndarray, 
    sigma: float, 
    last_sample: dict = None
) -> dimod.BinaryQuadraticModel:
    """
    Builds the QPHR Augmented Lagrangian BQM without ANY slack variables.
    L(x) = f(x) + (1 / 2*sigma) * sum( max(0, sigma*g_i(x) + lambda_i)^2 - lambda_i^2 )
    """
    bqm = dimod.BinaryQuadraticModel('BINARY')

    # Add original linear & quadratic terms
    for var, coeff in cqm.objective.linear.items():
        bqm.add_variable(var, float(coeff))

    for (v1, v2), coeff in cqm.objective.quadratic.items():
        bqm.add_interaction(v1, v2, float(coeff))

    bqm.offset += float(cqm.objective.offset)

    # Add PHR Augmented Penalties
    for i, constr in enumerate(parsed_constraints):
        lambda_i = lambda_vec[i]

        if last_sample is not None:
            g_i_val = evaluate_g_i(constr, last_sample)
            is_active = (lambda_i + sigma * g_i_val) > 0
        else:
            is_active = True

        if is_active:
            K = sigma * constr['b'] + lambda_i
            a_dict = constr['a']
            vars_list = list(a_dict.keys())

            # Linear update
            for var in vars_list:
                a_j = a_dict[var]
                C_j = sigma * a_j
                lin_bias = (1.0 / (2.0 * sigma)) * (C_j**2 + 2.0 * K * C_j)
                bqm.add_variable(var, lin_bias)

            # Quadratic coupling update
            for j in range(len(vars_list)):
                for k in range(j + 1, len(vars_list)):
                    v_j, v_k = vars_list[j], vars_list[k]
                    a_j, a_k = a_dict[v_j], a_dict[v_k]
                    quad_coupling = sigma * a_j * a_k
                    bqm.add_interaction(v_j, v_k, quad_coupling)

            constant_term = (K**2 - lambda_i**2) / (2.0 * sigma)
            bqm.offset += constant_term
        else:
            bqm.offset += (-(lambda_i**2) / (2.0 * sigma))

    return bqm


def compute_residual_R(parsed_constraints: list, sample: dict, lambda_vec: np.ndarray, sigma: float) -> float:
    """Computes constraint residual R_l according to Paper Eq. 16c."""
    sum_sq = 0.0
    for i, constr in enumerate(parsed_constraints):
        g_val = evaluate_g_i(constr, sample)
        val = max(-lambda_vec[i] / sigma, g_val)
        sum_sq += val**2
    return math.sqrt(sum_sq)