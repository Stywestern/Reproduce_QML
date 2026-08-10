############################################################################################
#                   PREPROCESSING: CQM TO BQM CONVERTER (STANDARD SLACK)
############################################################################################

import dimod

def cqm_to_bqm_slack(cqm: dimod.ConstrainedQuadraticModel, lagrange_mult: float = 3.0):
    """
    Translates a dimod.CQM into a dimod.BQM by converting inequality constraints into 
    slack variables with quadratic penalties.

    Returns:
        bqm (dimod.BinaryQuadraticModel): Penalty-embedded BQM.
        invert_mapping (dict): Variable inversion mapping.
        original_vars (list): List of original decision variable names (e.g., ['x1', ..., 'x6']).
        num_slacks (int): Count of slack variables generated.
    """
    bqm, invert_mapping = dimod.cqm_to_bqm(cqm, lagrange_multiplier=lagrange_mult)
    original_vars = list(cqm.variables)
    logical_qubits = len(bqm.variables)
    num_slacks = logical_qubits - len(original_vars)

    return bqm, invert_mapping, original_vars, num_slacks