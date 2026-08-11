############################################################################################
#                                       Imports
############################################################################################

# Third Party Libraries
import dimod

############################################################################################
#                                       Main Block
############################################################################################

class AppendixAModel:
    """
    Represents the synthetic Binary Integer Programming baseline model from Appendix A 
    of the QPHR-ALM paper.
    
    Variables:
        6 Binary Decision Variables (x1 to x6)
        
    Objective:
        Minimize f(x) = 6*x1 + 3*x2 - 5*x3 - 6*x4 + 4*x5 - 7*x6
        
    Constraints:
        c1: -2*x2 - 2*x5 - x6 <= -3
        c2: -x1 + x3 - x4 + 2*x6 <= 0
        c3: -x1 + x3 + x4 <= 0
    """
    
    def __init__(self):
        self.name = "Appendix A (Synthetic Binary Integer Baseline)"
        self.num_original_vars = 6
        self.known_optimum_cost = -4.0
        self.known_optimum_bitstring = "110101"
        self._cqm = None

    def build_cqm(self) -> dimod.ConstrainedQuadraticModel:
        """Builds and returns the Constrained Quadratic Model (CQM)."""
        cqm = dimod.ConstrainedQuadraticModel()
        
        # 1. Decision Variables
        x = [dimod.Binary(f"x{i+1}") for i in range(self.num_original_vars)]
        
        # 2. Objective Function: f(x)
        cqm.set_objective(6*x[0] + 3*x[1] - 5*x[2] - 6*x[3] + 4*x[4] - 7*x[5])
        
        # 3. Explicit Inequality Constraints (LHS <= RHS)
        cqm.add_constraint(-2*x[1] - 2*x[4] - x[5] <= -3, label='c1')
        cqm.add_constraint(-x[0] + x[2] - x[3] + 2*x[5] <= 0, label='c2')
        cqm.add_constraint(-x[0] + x[2] + x[3] <= 0, label='c3')
        
        self._cqm = cqm
        return cqm

    def get_cqm(self) -> dimod.ConstrainedQuadraticModel:
        """Returns the cached CQM model, building it if necessary."""
        if self._cqm is None:
            return self.build_cqm()
        return self._cqm


############################################################################################
#                                   Execution / Verification Block
############################################################################################
if __name__ == "__main__":
    print("--- TESTING PROBLEM MODULE: APPENDIX A ---")
    problem = AppendixAModel()
    model = problem.build_cqm()
    
    print(f"Problem Name : {problem.name}")
    print(f"Num Variables: {problem.num_original_vars}")
    print(f"Known Optimum: {problem.known_optimum_bitstring} (Cost: {problem.known_optimum_cost})\n")
    
    print("OBJECTIVE:")
    print(model.objective.to_polystring())
    
    print("\nCONSTRAINTS:")
    for label, constraint in model.constraints.items():
        print(f"[{label}]: {constraint.to_polystring()}")
        
    print("\nVARIABLES:")
    print(list(model.variables))