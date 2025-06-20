import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from scipy.stats import norm, expon, uniform

# Read the data
cost_df = pd.read_csv('https://raw.githubusercontent.com/Dhanvi199/schulich_data_science/refs/heads/main/cost.csv')
time_df = pd.read_csv('https://raw.githubusercontent.com/Dhanvi199/schulich_data_science/refs/heads/main/time.csv')

# Print the dataframes to verify their contents
print("\n=== Cost DataFrame ===")
print(cost_df)
print("\n=== Time DataFrame ===")
print(time_df)

# Define parameters
customers = list(range(1, 8))  # Customers 1-7
PENALTY_COST = 2000  # Cost per unit of shortage
DISPOSAL_COST = 5    # Cost per unit of disposal
MAX_TIME = 724       # Maximum total conversion time
MAX_INITIAL_PRODUCTION = 150000  # Maximum total initial production capacity

# Create cost and time matrices
cost_matrix = {}
time_matrix = {}
for i in range(7):
    for j in range(7):
        if i != j:
            cost_matrix[(i+1, j+1)] = cost_df.iloc[i, j+1]
            time_matrix[(i+1, j+1)] = time_df.iloc[i, j+1]

# Print the matrices to verify their contents
print("\n=== Cost Matrix ===")
for (i,j), cost in cost_matrix.items():
    print(f"Cost to convert from {i} to {j}: ${cost:,.2f}")

print("\n=== Time Matrix ===")
for (i,j), time in time_matrix.items():
    print(f"Time to convert from {i} to {j}: {time:,.2f} hours")

# Create the optimization model
model = gp.Model("Dye_Conversion")

# Decision Variables
# x[i,j] = amount of dye converted from type i to type j
x = model.addVars([(i, j) for i in customers for j in customers if i != j], lb=0, name="conversion")

# y[i] = amount of dye type i disposed
y = model.addVars(customers, lb=0, name="disposal")

# z[i] = amount of shortage for dye type i
z = model.addVars(customers, lb=0, name="shortage")

# Expected demand for each customer
expected_demand = {
    1: 15000,  # Normal(15000, 1000)
    2: 15000,  # Exponential(15000)
    3: 15000,  # Uniform(10000, 20000)
    4: 15000,  # Normal(15000, 500)
    5: 7500,   # Exponential(7500)
    6: 15000,  # Uniform(12000, 18000)
    7: 15000   # Normal(15000, 2000)
}

# Initial production amounts (to be determined)
initial_production = model.addVars(customers, lb=0, name="initial_production")

# Objective Function
# Minimize: Conversion costs + Disposal costs + Penalty costs
conversion_costs = gp.quicksum(cost_matrix[(i,j)] * x[i,j] for i in customers for j in customers if i != j)
disposal_costs = DISPOSAL_COST * gp.quicksum(y[i] for i in customers)
penalty_costs = PENALTY_COST * gp.quicksum(z[i] for i in customers)

model.setObjective(conversion_costs + disposal_costs + penalty_costs, GRB.MINIMIZE)

# Constraints

# 1. Time constraint
model.addConstr(
    gp.quicksum(time_matrix[(i,j)] * x[i,j] for i in customers for j in customers if i != j) <= MAX_TIME,
    name="time_limit"
)

# 2. Balance constraints for each dye type
for i in customers:
    # Initial production + Incoming conversions - Outgoing conversions - Disposal - Shortage = Expected demand
    model.addConstr(
        initial_production[i] +
        gp.quicksum(x[j,i] for j in customers if j != i) -
        gp.quicksum(x[i,j] for j in customers if j != i) -
        y[i] -
        z[i] == expected_demand[i],
        name=f"balance_{i}"
    )

# 3. Total initial production constraint
total_expected_demand = sum(expected_demand.values())
model.addConstr(
    gp.quicksum(initial_production[i] for i in customers) <= MAX_INITIAL_PRODUCTION,
    name="max_production"
)

# 4. Limit initial production per customer
for i in customers:
    model.addConstr(
        initial_production[i] <= 0.95 * expected_demand[i],  # Can't produce more than 95% of expected demand initially
        name=f"max_initial_{i}"
    )

# 5. Add a constraint to ensure we don't under-produce
model.addConstr(
    gp.quicksum(initial_production[i] for i in customers) >= 0.8 * total_expected_demand,  # Must produce at least 80% of expected demand
    name="min_production"
)

# 6. Add a constraint to ensure we don't over-produce any single dye type
for i in customers:
    model.addConstr(
        initial_production[i] + gp.quicksum(x[j,i] for j in customers if j != i) <= 1.2 * expected_demand[i],  # Can't have more than 120% of expected demand
        name=f"max_total_{i}"
    )

# Solve the model
try:
    model.optimize()
    
    if model.status == GRB.OPTIMAL:
        print("\n=== Optimization Results ===")
        print(f"Status: Optimal solution found")
        print(f"Total Cost: ${model.objVal:,.2f}")
        
        # Break down costs
        print("\n=== Cost Breakdown ===")
        print(f"Conversion Costs: ${conversion_costs.getValue():,.2f}")
        print(f"Disposal Costs: ${disposal_costs.getValue():,.2f}")
        print(f"Penalty Costs: ${penalty_costs.getValue():,.2f}")
        
        print("\n=== Initial Production ===")
        for i in customers:
            print(f"Customer {i}: {initial_production[i].x:,.2f} units")
        
        print("\n=== Conversions ===")
        conversions_made = False
        for i in customers:
            for j in customers:
                if i != j and x[i,j].x > 0.01:  # Only show significant conversions
                    conversions_made = True
                    print(f"Convert {x[i,j].x:,.2f} units from Customer {i} to Customer {j}")
        if not conversions_made:
            print("No conversions made")
        
        print("\n=== Disposals ===")
        disposals_made = False
        for i in customers:
            if y[i].x > 0.01:  # Only show significant disposals
                disposals_made = True
                print(f"Customer {i}: Dispose {y[i].x:,.2f} units")
        if not disposals_made:
            print("No disposals made")
        
        print("\n=== Shortages ===")
        shortages_exist = False
        for i in customers:
            if z[i].x > 0.01:  # Only show significant shortages
                shortages_exist = True
                print(f"Customer {i}: Shortage of {z[i].x:,.2f} units")
        if not shortages_exist:
            print("No shortages")
        
        print("\n=== Time Usage ===")
        total_time = sum(time_matrix[(i,j)] * x[i,j].x for i in customers for j in customers if i != j)
        print(f"Total conversion time used: {total_time:,.2f} hours")
        print(f"Maximum allowed time: {MAX_TIME:,.2f} hours")
        print(f"Time utilization: {(total_time/MAX_TIME)*100:.1f}%")
        
    else:
        print(f"\nNo optimal solution found. Status code: {model.status}")
        
except gp.GurobiError as e:
    print(f"Error code {e.errno}: {e}") 