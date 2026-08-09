############################################################################################
#                                           IMPORTS 
############################################################################################
import dimod


############################################################################################
#                                           Main Code
############################################################################################

def build_appendix_a_model():
    """Builds and returns the mathematical model for the Appendix A baseline."""

    print("[1] Initializing an empty Constrained Quadratic Model (CQM)...")
    cqm = dimod.ConstrainedQuadraticModel()
    
    # 1. Define variables
    print("[2] Creating 6 binary variables (x1 to x6)...")
    x = [dimod.Binary(f"x{i+1}") for i in range(6)]
    
    # 2. Define objective
    print("[3] Setting the objective function to minimize...")
    cqm.set_objective(6*x[0] + 3*x[1] - 5*x[2] - 6*x[3] + 4*x[4] - 7*x[5])
    
    # 3. Define constraints
    print("[4] Adding the three inequality constraints from the paper...")
    cqm.add_constraint(-2*x[1] - 2*x[4] - x[5] + 3 <= 0, label='c1')
    cqm.add_constraint(-x[0] + x[2] - x[3] + 2*x[5] <= 0, label='c2')
    cqm.add_constraint(-x[0] + x[2] + x[3] <= 0, label='c3')
    
    print("[5] Model construction complete.\n")
    print(cqm)
    return cqm

############################################################################################
#                               Execution Block
############################################################################################
if __name__ == "__main__":
    print("--- BUILDING APPENDIX A BASELINE ---")
    model = build_appendix_a_model()
    
    print("--- MODEL VERIFICATION ---")

    print("OBJECTIVE:")
    print(model.objective.to_polystring())
    
    print("\nCONSTRAINTS:")
    for label, constraint in model.constraints.items():
        print(f"[{label}]: {constraint.to_polystring()}")
        
    print("\nVARIABLES:")
    print(list(model.variables))
