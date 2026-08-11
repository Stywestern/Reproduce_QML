############################################################################################
#                                       Imports
############################################################################################

# Third Party Libraries
import dimod
import pulp

############################################################################################
#                                       Main Block
############################################################################################

def cqm_to_pulp(cqm: dimod.ConstrainedQuadraticModel) -> tuple[pulp.LpProblem, dict]:
    """
    Translates a dimod.CQM object into a native PuLP LpProblem representation.
    Supports both Binary decision variables and Continuous Real variables.
    
    Returns:
        prob (pulp.LpProblem): The configured PuLP problem instance.
        pulp_vars (dict): Mapping from variable names to pulp.LpVariable instances.
    """
    prob = pulp.LpProblem("CQM_Converted_Model", pulp.LpMinimize)

    # 1. Register Variables (Binary + Real / Continuous)
    pulp_vars = {}
    for var in cqm.variables:
        vartype = cqm.vartype(var)
        var_str = str(var)

        if vartype == dimod.BINARY:
            pulp_vars[var_str] = pulp.LpVariable(var_str, cat=pulp.LpBinary)
        elif vartype == dimod.SPIN:
            # Map Ising spin {-1, +1} to binary {0, 1}
            pulp_vars[var_str] = pulp.LpVariable(var_str, cat=pulp.LpBinary)
        elif vartype == dimod.REAL:
            lb = cqm.lower_bound(var)
            ub = cqm.upper_bound(var)
            pulp_vars[var_str] = pulp.LpVariable(
                var_str, 
                lowBound=lb, 
                upBound=ub, 
                cat=pulp.LpContinuous
            )
        else:
            raise ValueError(f"Unsupported vartype '{vartype}' for variable '{var}'.")

    # 2. Objective Function Construction
    if cqm.objective.quadratic:
        raise ValueError("CQM model contains quadratic objective terms! CBC linear solver requires linear objectives.")

    obj_terms = [bias * pulp_vars[str(var)] for var, bias in cqm.objective.linear.items()]
    obj_expr = pulp.lpSum(obj_terms) + cqm.objective.offset
    prob += obj_expr

    # 3. Constraints Construction
    for label, constraint in cqm.constraints.items():
        if constraint.lhs.quadratic:
            raise ValueError(f"Constraint '{label}' contains quadratic terms! CBC requires linear constraints.")

        # Build linear expression for LHS
        lhs_terms = [bias * pulp_vars[str(var)] for var, bias in constraint.lhs.linear.items()]
        lhs_expr = pulp.lpSum(lhs_terms)

        # Move numerical offset to RHS: lhs_expr + offset <= 0  ==>  lhs_expr <= -offset
        rhs = -constraint.lhs.offset
        sense_str = str(constraint.sense)

        if "<=" in sense_str or constraint.sense == dimod.Sense.Le:
            prob += (lhs_expr <= rhs, str(label))
        elif ">=" in sense_str or constraint.sense == dimod.Sense.Ge:
            prob += (lhs_expr >= rhs, str(label))
        elif "==" in sense_str or constraint.sense == dimod.Sense.Eq:
            prob += (lhs_expr == rhs, str(label))

    return prob, pulp_vars