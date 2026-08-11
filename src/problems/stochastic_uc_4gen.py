############################################################################################
#                                       Imports
############################################################################################

# Third Party Libraries
import numpy as np
import dimod

############################################################################################
#                                       Main Block
############################################################################################

class FourGeneratorUCModel:
    """
    Implements the 4-Generator Stochastic Unit Commitment (SUC) problem from the paper.
    Incorporates Weibull wind generation and Beta demand distributions across a 24-hour horizon.
    Uses a hybrid two-stage formulation:
      - Binary Commitment: u_{i,t} in {0, 1} (Master Problem on QPU)
      - Continuous Generation: P_{i,t} in [P_min, P_max] (Subproblem on LP)
    """

    def __init__(self, num_hours: int = 24, num_scenarios: int = 3, seed: int = 42):
        self.num_hours = num_hours
        self.num_scenarios = num_scenarios
        self.seed = seed

        # Generator parameters: [Gen 1, Gen 2, Gen 3, Gen 4]
        self.gen_names = ["G1_Coal", "G2_CCGT", "G3_GasPeaker", "G4_FastPeaker"]
        self.num_gens = len(self.gen_names)

        # Operational limits (MW)
        self.p_min = np.array([20.0, 15.0, 10.0, 5.0])
        self.p_max = np.array([150.0, 100.0, 80.0, 50.0])

        # Linear Dispatch/Variable Cost coefficients ($/MWh)
        self.c_var = np.array([15.0, 25.0, 30.0, 50.0])

        # Startup / Fixed Hourly Commitment Costs ($/hr)
        self.c_fixed = np.array([500.0, 300.0, 200.0, 100.0])

        # Generate stochastic wind and demand profile matrices
        self.demand, self.wind = self._generate_stochastic_profiles()

    def _generate_stochastic_profiles(self):
        """Generates Weibull wind speeds and Beta demand profiles."""
        np.random.seed(self.seed)

        # 1. Base Diurnal Demand Profile (24 Hours)
        time_steps = np.arange(self.num_hours)
        base_demand = 180 + 80 * np.sin((time_steps - 6) * np.pi / 12)  # Peaks around hour 18

        # 2. Beta Distribution Noise for Demand Scenarios
        demand_scenarios = np.zeros((self.num_scenarios, self.num_hours))
        for s in range(self.num_scenarios):
            noise = np.random.beta(a=2.0, b=5.0, size=self.num_hours) * 40.0 - 15.0
            demand_scenarios[s] = base_demand + noise

        # 3. Weibull Distribution for Wind Power Generation
        wind_scenarios = np.zeros((self.num_scenarios, self.num_hours))
        for s in range(self.num_scenarios):
            # Weibull wind speed (shape k=2.0, scale c=10.0)
            wind_speed = np.random.weibull(a=2.0, size=self.num_hours) * 10.0
            # Cut-in / rated wind power mapping (Max 60 MW wind farm)
            wind_power = np.clip((wind_speed / 12.0) ** 3 * 60.0, 0.0, 60.0)
            wind_scenarios[s] = wind_power

        return demand_scenarios, wind_scenarios

    def get_cqm(self) -> dimod.ConstrainedQuadraticModel:
        """
        Builds the master Unit Commitment Constrained Quadratic Model (CQM).
        Binary variables u_{i,t} in {0, 1} indicate generator ON/OFF status at hour t.
        Continuous variables P_{i,t} represent generator power output in MW.
        """
        cqm = dimod.ConstrainedQuadraticModel()

        # 1. Register Decision Variables
        u = {}  # Binary commitment: u_{i,t}
        P = {}  # Continuous dispatch: P_{i,t}

        for t in range(self.num_hours):
            for i, gen in enumerate(self.gen_names):
                u_name = f"u_{gen}_t{t}"
                p_name = f"P_{gen}_t{t}"

                u[i, t] = dimod.Binary(u_name)
                P[i, t] = dimod.Real(p_name, lower_bound=0.0, upper_bound=self.p_max[i])

        # 2. Objective Function: Total Cost = Fixed Commitment Cost + Variable Dispatch Cost
        total_cost = 0
        for t in range(self.num_hours):
            for i in range(self.num_gens):
                total_cost += (self.c_fixed[i] * u[i, t]) + (self.c_var[i] * P[i, t])

        cqm.set_objective(total_cost)

        # 3. Add Generator Coupling Limits: P_min * u_{i,t} <= P_{i,t} <= P_max * u_{i,t}
        for t in range(self.num_hours):
            for i, gen in enumerate(self.gen_names):
                cqm.add_constraint(
                    P[i, t] - (self.p_max[i] * u[i, t]) <= 0,
                    label=f"p_max_coupling_{gen}_t{t}"
                )
                cqm.add_constraint(
                    (self.p_min[i] * u[i, t]) - P[i, t] <= 0,
                    label=f"p_min_coupling_{gen}_t{t}"
                )

        # 4. Add Master Power Balance Constraints across all stochastic scenarios
        # For each hour t and scenario s:
        # Sum(P_{i,t}) + Wind[s, t] >= Demand[s, t]
        for t in range(self.num_hours):
            for s in range(self.num_scenarios):
                net_demand = self.demand[s, t] - self.wind[s, t]
                total_generation_expr = sum(P[i, t] for i in range(self.num_gens))

                # Enforce power balance constraint: Net Demand - Generation <= 0
                cqm.add_constraint(
                    net_demand - total_generation_expr <= 0,
                    label=f"power_balance_t{t}_s{s}"
                )

        return cqm


############################################################################################
#                                  Execution Block
############################################################################################
if __name__ == "__main__":
    print("==========================================================================")
    print("        STOCHASTIC 4-GENERATOR UNIT COMMITMENT MODEL INSPECTION           ")
    print("==========================================================================")

    # 1. Instantiate Problem & Build CQM
    problem = FourGeneratorUCModel(num_hours=24, num_scenarios=3, seed=42)
    cqm = problem.get_cqm()

    binary_vars = [v for v in cqm.variables if cqm.vartype(v) == dimod.BINARY]
    real_vars = [v for v in cqm.variables if cqm.vartype(v) == dimod.REAL]
    num_constraints = len(cqm.constraints)

    # 2. General Overview
    print("\n1. MODEL DIMENSION & STRUCTURE")
    print(f"  -> Problem Class:                Mixed-Integer Linear Program (MILP / CQM)")
    print(f"  -> Total Decision Variables:     {len(cqm.variables)}")
    print(f"       * Binary Qubits (u_{{i,t}}):    {len(binary_vars)} (4 generators * 24 hours)")
    print(f"       * Continuous MW (P_{{i,t}}):     {len(real_vars)} (4 generators * 24 hours)")
    print(f"  -> Total Constraints:            {num_constraints}")
    print(f"       * Coupling Limits (P_min/max): {4 * 24 * 2} constraints")
    print(f"       * Power Balance (Scenarios):   {3 * 24} constraints (3 scenarios * 24 hours)")

    # 3. Generator Parameters
    print("\n2. THERMAL GENERATOR PARAMETERS")
    print(f"  {'Gen Name':<15} | {'Fixed Cost ($/hr)':<18} | {'Var Cost ($/MWh)':<16} | {'Min (MW)':<8} | {'Max (MW)':<8}")
    print("  " + "-" * 75)
    for i, gen in enumerate(problem.gen_names):
        print(f"  {gen:<15} | ${problem.c_fixed[i]:<17.2f} | ${problem.c_var[i]:<15.2f} | {problem.p_min[i]:<8.1f} | {problem.p_max[i]:<8.1f}")

    # 4. Stochastic Scenario Profiles Sample
    print("\n3. STOCHASTIC SCENARIO PROFILES SAMPLE (Demand & Wind Power)")
    print(f"  {'Hour':<6} | {'Scen 1 Dem (MW)':<16} | {'Scen 1 Wind (MW)':<17} | {'Scen 1 Net Dem (MW)':<20}")
    print("  " + "-" * 68)
    sample_hours = [0, 6, 12, 18, 23]  # Night, Morning, Noon, Peak Evening, Night
    for t in sample_hours:
        dem = problem.demand[0, t]
        wnd = problem.wind[0, t]
        net_dem = dem - wnd
        print(f"  {t:<6} | {dem:<16.2f} | {wnd:<17.2f} | {net_dem:<20.2f}")

    # 5. Sample Variable Names
    print("\n4. SAMPLE DECISION VARIABLES (Registered in CQM)")
    print("  -> Binary Qubit Variables (Stage 1):")
    print(f"       {binary_vars[:4]}")
    print("  -> Continuous Dispatch Variables (Stage 2):")
    print(f"       {real_vars[:4]}")

    # 6. Sample Equations Printout
    print("\n5. SAMPLE MATHEMATICAL EQUATIONS IN MODEL")
    print("  -> Sample Coupling Constraint (G1 at Hour 0):")
    sample_label_coupling = "p_max_coupling_G1_Coal_t0"
    print(f"       {cqm.constraints[sample_label_coupling].to_polystring()}")

    print("  -> Sample Power Balance Constraint (Hour 18 Peak, Scenario 0):")
    sample_label_balance = "power_balance_t18_s0"
    print(f"       {cqm.constraints[sample_label_balance].to_polystring()}")

    print("\n==========================================================================")
    print("               MODEL BLUEPRINT CREATED SUCCESSFULLY                       ")
    print("==========================================================================")