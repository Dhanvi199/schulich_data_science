import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from scipy.stats import norm, expon, uniform
import time

def generate_demand_scenario():
    """Generate a single scenario of demands for all customers"""
    return {
        1: norm(10000, 2000).rvs(),    # Customer 1: Normal distribution
        2: expon(scale=15000).rvs(),   # Customer 2: Exponential distribution
        3: uniform(5000, 15000).rvs(), # Customer 3: Uniform distribution
        4: norm(12000, 3000).rvs(),    # Customer 4: Normal distribution
        5: expon(scale=20000).rvs(),   # Customer 5: Exponential distribution
        6: uniform(8000, 20000).rvs(), # Customer 6: Uniform distribution
        7: norm(15000, 4000).rvs()     # Customer 7: Normal distribution
    }

def solve_optimization(demands):
    """Solve the optimization problem for a given demand scenario"""
    # Read the data
    cost_df = pd.read_csv('https://raw.githubusercontent.com/Dhanvi199/schulich_data_science/refs/heads/main/cost.csv', index_col=0)
    time_df = pd.read_csv('https://raw.githubusercontent.com/Dhanvi199/schulich_data_science/refs/heads/main/time.csv', index_col=0)

    # Define parameters
    customers = list(range(1, 8))
    PENALTY_COST = 2000
    DISPOSAL_COST = 5
    MAX_TIME = 724
    INITIAL_PRODUCTION_COST = 1000

    # Create cost and time matrices
    cost_matrix = {}
    time_matrix = {}
    for i in range(7):
        for j in range(7):
            if i != j:
                cost_matrix[(i+1, j+1)] = float(cost_df.iloc[i, j])
                time_matrix[(i+1, j+1)] = float(time_df.iloc[i, j])

    # Create the model
    model = gp.Model("Dye_Conversion_Optimization")

    # Decision variables
    conversions = model.addVars(cost_matrix.keys(), name="conversions")
    disposals = model.addVars(customers, name="disposals")
    shortages = model.addVars(customers, name="shortages")
    initial_production = model.addVars(customers, name="initial_production")

    # Objective function
    total_cost = (
        gp.quicksum(conversions[i,j] * cost_matrix[i,j] for i,j in cost_matrix.keys()) +
        gp.quicksum(disposals[i] * DISPOSAL_COST for i in customers) +
        gp.quicksum(shortages[i] * PENALTY_COST for i in customers) +
        gp.quicksum(initial_production[i] * INITIAL_PRODUCTION_COST for i in customers)
    )
    model.setObjective(total_cost, GRB.MINIMIZE)

    # Time constraint
    model.addConstr(
        gp.quicksum(conversions[i,j] * time_matrix[i,j] for i,j in time_matrix.keys()) <= MAX_TIME,
        "time_limit"
    )

    # Balance constraints for each customer
    for i in customers:
        model.addConstr(
            initial_production[i] + 
            gp.quicksum(conversions[j,i] for j in range(1,8) if (j,i) in conversions) -
            gp.quicksum(conversions[i,j] for j in range(1,8) if (i,j) in conversions) -
            disposals[i] + shortages[i] == demands[i],
            f"balance_{i}"
        )

    # Initial production limits
    for i in customers:
        model.addConstr(initial_production[i] <= 0.95 * demands[i], f"prod_limit_{i}")

    # Minimum conversion requirement
    model.addConstr(
        gp.quicksum(conversions[i,j] for i,j in conversions.keys()) >= 300,
        "min_conversion"
    )

    # Optimize
    model.optimize()

    if model.status == GRB.OPTIMAL:
        return model.objVal
    else:
        return None

def run_monte_carlo_simulation(num_trials=50, scenarios_per_trial=100):
    """Run Monte Carlo simulation with SAA"""
    all_costs = []
    
    for trial in range(num_trials):
        print(f"\nRunning trial {trial + 1}/{num_trials}")
        trial_costs = []
        
        for scenario in range(scenarios_per_trial):
            demands = generate_demand_scenario()
            cost = solve_optimization(demands)
            if cost is not None:
                trial_costs.append(cost)
        
        if trial_costs:
            avg_cost = np.mean(trial_costs)
            all_costs.append(avg_cost)
            print(f"Trial {trial + 1} average cost: ${avg_cost:,.2f}")
    
    return all_costs

def calculate_confidence_interval(costs, confidence=0.95):
    """Calculate confidence interval for the costs"""
    mean_cost = np.mean(costs)
    std_error = np.std(costs) / np.sqrt(len(costs))
    z_score = norm.ppf((1 + confidence) / 2)
    margin_of_error = z_score * std_error
    
    return mean_cost, (mean_cost - margin_of_error, mean_cost + margin_of_error)

if __name__ == "__main__":
    print("Starting Monte Carlo simulation...")
    start_time = time.time()
    
    costs = run_monte_carlo_simulation()
    mean_cost, ci = calculate_confidence_interval(costs)
    
    print("\nFinal Results:")
    print(f"Number of successful trials: {len(costs)}")
    print(f"Optimal Expected Cost: ${mean_cost:,.2f}")
    print(f"95% Confidence Interval: [${ci[0]:,.2f}, ${ci[1]:,.2f}]")
    print(f"Total simulation time: {time.time() - start_time:.2f} seconds") 