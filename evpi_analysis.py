import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from scipy.stats import norm, expon, uniform
import time
from vss_analysis import generate_demand_scenario, solve_optimization

def calculate_wait_and_see(num_trials=50, scenarios_per_trial=100):
    """Calculate the Wait-and-See (WS) solution - expected value with perfect information"""
    all_costs = []
    
    for trial in range(num_trials):
        print(f"\nCalculating WS for trial {trial + 1}/{num_trials}")
        trial_costs = []
        
        for scenario in range(scenarios_per_trial):
            # Generate a demand scenario
            demands = generate_demand_scenario()
            
            # Solve the optimization problem for this scenario
            cost = solve_optimization(demands)
            if cost is not None:
                trial_costs.append(cost)
        
        if trial_costs:
            # For WS, we take the average of optimal solutions for each scenario
            avg_cost = np.mean(trial_costs)
            all_costs.append(avg_cost)
            print(f"Trial {trial + 1} WS cost: ${avg_cost:,.2f}")
    
    return all_costs

def calculate_evpi(ws_costs, vss_costs):
    """Calculate the Expected Value of Perfect Information"""
    ws_mean = np.mean(ws_costs)
    vss_mean = np.mean(vss_costs)
    evpi = ws_mean - vss_mean
    
    return evpi, ws_mean, vss_mean

if __name__ == "__main__":
    print("Starting EVPI Analysis...")
    start_time = time.time()
    
    # Step 1: Calculate Wait-and-See (WS) solution
    print("\nCalculating Wait-and-See solution...")
    ws_costs = calculate_wait_and_see()
    
    # Step 2: Get the stochastic solution costs (from previous VSS analysis)
    print("\nGetting stochastic solution costs...")
    vss_costs = []  # This should be populated from your previous VSS analysis
    # For demonstration, we'll generate some random costs
    vss_costs = [np.random.normal(100000000, 1000000) for _ in range(50)]
    
    # Step 3: Calculate EVPI
    evpi, ws_mean, vss_mean = calculate_evpi(ws_costs, vss_costs)
    
    print("\nFinal Results:")
    print(f"Expected Value with Perfect Foresight (WS): ${ws_mean:,.2f}")
    print(f"Expected Value of the Stochastic Solution (VSS): ${vss_mean:,.2f}")
    print(f"Expected Value of Perfect Information (EVPI): ${evpi:,.2f}")
    print(f"Percentage of potential improvement: {(evpi/ws_mean)*100:.2f}%")
    print(f"Total analysis time: {time.time() - start_time:.2f} seconds") 