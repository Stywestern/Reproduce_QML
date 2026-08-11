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
    prob = pulp.LpProblem("CQM_Converted_Model", pulp.LpMinimize)

    # 1. Register Variables & Track Spin Mappings
    pulp_vars = {}
    is_spin_var = {}

    for var in cqm.variables:
        vartype = cqm.vartype(var)
        var_str = str(var)

        if vartype == dimod.BINARY:
            pulp_vars[var_str] = pulp.LpVariable(var_str, cat=pulp.LpBinary)
            is_spin_var[var_str] = False
        elif vartype == dimod.SPIN:
            pulp_vars[var_str] = pulp.LpVariable(var_str, cat=pulp.LpBinary)
            is_spin_var[var_str] = True
        elif vartype == dimod.REAL:
            lb = cqm.lower_bound(var)
            ub = cqm.upper_bound(var)
            pulp_vars[var_str] = pulp.LpVariable(var_str, lowBound=lb, upBound=ub, cat=pulp.LpContinuous)
            is_spin_var[var_str] = False
        else:
            raise ValueError(f"Unsupported vartype '{vartype}' for variable '{var}'.")

    # 2. Build Objective Function
    obj_expr = pulp.LpAffineExpression()
    running_offset = cqm.objective.offset

    for var, bias in cqm.objective.linear.items():
        var_str = str(var)
        if is_spin_var[var_str]:
            obj_expr += (2.0 * bias) * pulp_vars[var_str]
            running_offset -= bias
        else:
            obj_expr += bias * pulp_vars[var_str]

    prob += obj_expr + running_offset

    # 3. Build Constraints (THE FIX IS HERE)
    for label, constraint in cqm.constraints.items():
        lhs_expr = pulp.LpAffineExpression()
        
        # safely extract RHS (defaults to 0.0 if not present) and subtract the offset
        rhs_val = getattr(constraint, 'rhs', 0.0)
        running_rhs = rhs_val - constraint.lhs.offset

        for var, bias in constraint.lhs.linear.items():
            var_str = str(var)
            if is_spin_var[var_str]:
                lhs_expr += (2.0 * bias) * pulp_vars[var_str]
                running_rhs += bias
            else:
                lhs_expr += bias * pulp_vars[var_str]

        sense_str = str(getattr(constraint.sense, 'value', constraint.sense))

        if "<=" in sense_str or "Le" in sense_str:
            prob += (lhs_expr <= running_rhs, str(label))
        elif ">=" in sense_str or "Ge" in sense_str:
            prob += (lhs_expr >= running_rhs, str(label))
        elif "==" in sense_str or "Eq" in sense_str:
            prob += (lhs_expr == running_rhs, str(label))
        else:
            raise ValueError(f"Unrecognized constraint sense '{sense_str}' on constraint '{label}'")

    return prob, pulp_vars