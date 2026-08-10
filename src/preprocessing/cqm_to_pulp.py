############################################################################################
#                        TRANSFORMERS: CQM TO PULP CONVERTER
############################################################################################

import dimod
import pulp


def cqm_to_pulp(cqm: dimod.ConstrainedQuadraticModel) -> tuple[pulp.LpProblem, dict]:
    """
    Translates a dimod.CQM object into a native PuLP LpProblem representation.
    
    Returns:
        prob (pulp.LpProblem): The configured PuLP problem instance.
        pulp_vars (dict): Mapping from variable names to pulp.LpVariable instances.
    """
    prob = pulp.LpProblem("CQM_Converted_Model", pulp.LpMinimize)

    # 1. Variables
    pulp_vars = {}
    for var in cqm.variables:
        if cqm.vartype(var) != dimod.BINARY:
            raise ValueError(f"Variable '{var}' is not BINARY. Linear PuLP transformer expects binary models.")
        pulp_vars[var] = pulp.LpVariable(var, cat=pulp.LpBinary)

    # 2. Objective Function
    if cqm.objective.quadratic:
        raise ValueError("CQM model contains quadratic objective terms! PuLP transformer requires linear objectives.")

    obj_expr = sum(bias * pulp_vars[var] for var, bias in cqm.objective.linear.items())
    obj_expr += cqm.objective.offset
    prob += obj_expr

    # 3. Constraints
    for label, constraint in cqm.constraints.items():
        if constraint.lhs.quadratic:
            raise ValueError(f"Constraint '{label}' contains quadratic terms! PuLP requires linear constraints.")

        lhs_expr = sum(bias * pulp_vars[var] for var, bias in constraint.lhs.linear.items())
        lhs_expr += constraint.lhs.offset

        rhs = constraint.rhs
        sense = constraint.sense.value

        if sense == '<=':
            prob += (lhs_expr <= rhs, label)
        elif sense == '>=':
            prob += (lhs_expr >= rhs, label)
        elif sense == '==':
            prob += (lhs_expr == rhs, label)

    return prob, pulp_vars