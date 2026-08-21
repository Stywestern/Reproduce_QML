# ###########################################################################################################################################
# Imports
# ###########################################################################################################################################
from src.problems.suc_problem_generator import SUCProblemData
import numpy as np

# ###########################################################################################################################################
# Realistic Problems
# ###########################################################################################################################################
def g4t24s3_suc_problem(seed=2):
    num_hours = 24
    num_scenarios = 3
    
    rng = np.random.default_rng(seed)
    t = np.arange(num_hours)

    base_demand = 220 + 90 * np.sin((t - 6) * np.pi / 12)
    custom_demand = np.array([
        base_demand + rng.normal(0, 10, num_hours)
        for _ in range(num_scenarios)
    ])

    custom_wind = np.array([
        np.clip(rng.normal(20, 15, num_hours), 0, 40)
        for _ in range(num_scenarios)
    ])

    pd = SUCProblemData.generate(
        num_gens=4,
        num_hours=num_hours,
        num_scenarios=num_scenarios,
        seed=seed,
        p_max=np.array([100.0, 95.0, 80.0, 75.0]),
        p_min=np.array([40.0, 35.0, 30.0, 25.0]),
        c_var=np.array([25.0, 30.0, 40.0, 60.0]),
        c_fixed=np.array([300.0, 240.0, 200.0, 180.0]),
        min_up=np.array([10, 8, 8, 6]),
        min_down=np.array([10, 8, 8, 6]),
        scenario_probs=np.array([0.2, 0.5, 0.3]),
        demand=custom_demand,
        wind=custom_wind
    )

    return pd, "g4t24s3"

def g10t24s10_suc_problem(seed=2):
    num_gens = 10
    num_hours = 24
    num_scenarios = 10
    
    rng = np.random.default_rng(seed)
    t = np.arange(num_hours)

    # Scale demand to match new system capacity (~1020 MW total)
    base_demand = 650 + 250 * np.sin((t - 6) * np.pi / 12)
    custom_demand = np.array([
        base_demand + rng.normal(0, 30, num_hours)
        for _ in range(num_scenarios)
    ])

    # High variance wind profiles to ensure scenarios require distinctly different recourses
    custom_wind = np.array([
        np.clip(rng.normal(80, 40, num_hours), 0, 200)
        for _ in range(num_scenarios)
    ])
    
    # Generate and normalize random scenario probabilities
    raw_probs = rng.uniform(0.5, 1.5, num_scenarios)
    scenario_probs = raw_probs / raw_probs.sum()

    pd = SUCProblemData.generate(
        num_gens=num_gens,
        num_hours=num_hours,
        num_scenarios=num_scenarios,
        seed=seed,
        
        # Mix of massive baseload, mid-merit, and fast peaker units
        p_max=np.array([250.0, 200.0, 150.0, 120.0, 100.0, 80.0, 50.0, 40.0, 20.0, 10.0]),
        p_min=np.array([100.0,  80.0,  50.0,  40.0,  30.0, 25.0, 10.0, 10.0,  0.0,  0.0]),
        
        c_var=np.array([ 15.0,  18.0,  22.0,  25.0,  30.0, 35.0, 45.0, 50.0, 70.0, 90.0]),
        c_fixed=np.array([1000.0, 800.0, 600.0, 500.0, 400.0, 300.0, 200.0, 150.0, 50.0, 20.0]),
        
        min_up=np.array([12, 10, 8, 8, 6, 5, 4, 3, 1, 1]),
        min_down=np.array([10, 8, 8, 6, 6, 5, 4, 3, 1, 1]),
        
        scenario_probs=scenario_probs,
        demand=custom_demand,
        wind=custom_wind,
        shed_cost=300.0, 
        spill_cost=50.0
    )

    return pd, "g10t24s10"

# ###########################################################################################################################################
# Mock Problems
# ###########################################################################################################################################

def tiny_suc_problem(seed=2):
    num_gens = 2
    num_hours = 3
    num_scenarios = 2

    pd = SUCProblemData.generate(
        num_gens=num_gens,
        num_hours=num_hours,
        num_scenarios=num_scenarios,
        seed=seed,

        p_max=np.array([100.0, 60.0]),
        p_min=np.array([20.0, 10.0]),

        c_var=np.array([10.0, 30.0]),
        c_fixed=np.array([50.0, 20.0]),

        # Disable these for the first test
        min_up=np.array([1, 1]),
        min_down=np.array([1, 1]),

        scenario_probs=np.array([0.5, 0.5]),

        demand=np.array([
            [70.0, 80.0, 60.0],   # scenario 0
            [90.0, 50.0, 80.0],   # scenario 1
        ]),

        wind=np.zeros((2, 3))
    )

    return pd, "tiny"

def long_horizon_suc_problem(seed=2):
    num_gens = 2
    num_hours = 96
    num_scenarios = 2

    pd = SUCProblemData.generate(
        num_gens=num_gens,
        num_hours=num_hours,
        num_scenarios=num_scenarios,
        seed=seed,

        p_max=np.array([100., 95.]),
        p_min=np.array([40., 35.]),
        c_var=np.array([25., 30.]),
        c_fixed=np.array([300., 240.]),

        min_up=np.array([1, 1]),
        min_down=np.array([1, 1]),

        scenario_probs=np.array([0.4, 0.6]),

        demand=np.array([
                [70.0, 80.0, 60.0],   # scenario 0
                [90.0, 50.0, 80.0],   # scenario 1
        ]),
        
        wind=np.zeros((2, 3))
    )

    return pd, "long"

def many_scenario_suc_problem(seed=2):
    num_gens = 2
    num_hours = 24
    num_scenarios = 10

    pd = SUCProblemData.generate(
        num_gens=num_gens,
        num_hours=num_hours,
        num_scenarios=num_scenarios,
        seed=seed,

        p_max=np.array([100., 95.]),
        p_min=np.array([40., 35.]),
        c_var=np.array([25., 30.]),
        c_fixed=np.array([300., 240.]),

        min_up=np.array([1, 1]),
        min_down=np.array([1, 1]),

        scenario_probs=np.full(num_scenarios, 1.0 / num_scenarios),

        demand=np.array([
            [70.0, 80.0, 60.0],   # scenario 0
            [90.0, 50.0, 80.0],   # scenario 1
            [75.0, 85.0, 65.0],   # scenario 2
            [85.0, 70.0, 75.0],   # scenario 3
            [65.0, 90.0, 70.0],   # scenario 4
            [80.0, 75.0, 85.0],   # scenario 5
            [95.0, 60.0, 90.0],   # scenario 6
            [70.0, 95.0, 75.0],   # scenario 7
            [85.0, 85.0, 65.0],   # scenario 8
            [75.0, 65.0, 90.0],   # scenario 9
        ]),

        wind=np.zeros((10, 3))
    )

    return pd, "varied"