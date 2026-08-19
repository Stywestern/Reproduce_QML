from src.problems.suc_problem_generator import SUCProblemData
import numpy as np

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

    return pd

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

    return pd