import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from itertools import combinations

# Read the data from GitHub
df = pd.read_csv('https://raw.githubusercontent.com/Dhanvi199/schulich_data_science/refs/heads/main/delivery.csv')

# Extract the data
customers = list(range(1, 16))  # Customers 1-15
vans = list(range(1, 4))  # 3 vans
weights = df['Size'].values
depot_distances = df['Depot'].values

# Create distance matrix
distances = {}
for i in range(15):
    for j in range(15):
        if i != j:
            distances[(i+1, j+1)] = df.iloc[i, j+3]

def find_subtours(solution, van_k):
    """Find all subtours in the current solution for a specific van"""
    subtours = []
    # Get all customers assigned to this van
    assigned_customers = [i for i in customers if any(solution.get((i,j), 0) > 0.5 or solution.get((j,i), 0) > 0.5 for j in customers if i != j)]
    
    if not assigned_customers:
        return []
        
    # Start with first assigned customer
    unvisited = set(assigned_customers)
    while unvisited:
        start = unvisited.pop()
        subtour = [start]
        current = start
        
        # Follow the route
        while True:
            next_customer = None
            for j in unvisited:
                if solution.get((current, j), 0) > 0.5 or solution.get((j, current), 0) > 0.5:
                    next_customer = j
                    break
            
            if next_customer is None:
                break
                
            subtour.append(next_customer)
            current = next_customer
            unvisited.discard(next_customer)
        
        if len(subtour) > 1 and len(subtour) < len(assigned_customers):
            subtours.append(subtour)
    
    return subtours

def solve_with_subtour_elimination():
    # Create the optimization model
    model = gp.Model("FedEx_Routing_With_Subtours")

    # Decision Variables
    x = model.addVars([(i, j, k) for i in customers for j in customers for k in vans if i != j], vtype=GRB.BINARY, name="route")
    y = model.addVars([(i, k) for i in customers for k in vans], vtype=GRB.BINARY, name="assignment")
    z = model.addVars(vans, lb=0, name="weight")
    max_weight = model.addVar(lb=0, name="max_weight")
    min_weight = model.addVar(lb=0, name="min_weight")

    # Objective Function
    model.setObjective(max_weight - min_weight, GRB.MINIMIZE)

    # Basic Constraints
    # 1. Each customer must be assigned to exactly one van
    for i in customers:
        model.addConstr(gp.quicksum(y[i,k] for k in vans) == 1)

    # 2. All vans must be used
    for k in vans:
        model.addConstr(gp.quicksum(y[i,k] for i in customers) >= 1)

    # 3. Weight constraints for each van
    for k in vans:
        model.addConstr(z[k] == gp.quicksum(weights[i-1] * y[i,k] for i in customers))
        model.addConstr(z[k] <= 15000)  # Maximum capacity
        model.addConstr(max_weight >= z[k])
        model.addConstr(min_weight <= z[k])

    # 4. Distance constraints for each van
    for k in vans:
        model.addConstr(
            gp.quicksum(depot_distances[i-1] * y[i,k] for i in customers) +
            gp.quicksum(distances[(i,j)] * x[i,j,k] for i in customers for j in customers if i != j) <= 254
        )

    # 5. Flow conservation constraints
    for k in vans:
        for i in customers:
            model.addConstr(gp.quicksum(x[i,j,k] for j in customers if i != j) == y[i,k])
            model.addConstr(gp.quicksum(x[j,i,k] for j in customers if i != j) == y[i,k])

    # 6. Special constraints
    # No more than 2 customers from {7,8,9} on same van
    for k in vans:
        model.addConstr(gp.quicksum(y[i,k] for i in [7,8,9]) <= 2)

    # Customers 10,11,12 must be on same van
    for k in vans:
        model.addConstr(y[10,k] == y[11,k])
        model.addConstr(y[11,k] == y[12,k])

    # If customer 1 is assigned, at least one of 13 or 14 must be on that van
    for k in vans:
        model.addConstr(y[1,k] <= y[13,k] + y[14,k])

    # Customer 2 cannot be with 3,4,5
    for k in vans:
        model.addConstr(y[2,k] + y[3,k] <= 1)
        model.addConstr(y[2,k] + y[4,k] <= 1)
        model.addConstr(y[2,k] + y[5,k] <= 1)

    # Maximum 5 deliveries per van
    for k in vans:
        model.addConstr(gp.quicksum(y[i,k] for i in customers) <= 5)

    # Iterative subtour elimination
    subtour_constraints = []
    iteration = 0
    max_iterations = 100  # Prevent infinite loop
    
    while iteration < max_iterations:
        model.optimize()
        
        if model.status != GRB.OPTIMAL:
            print("Model failed to solve optimally")
            break
            
        # Check for subtours in each van's route
        subtours_found = False
        for k in vans:
            # Get the current solution for this van
            current_solution = {(i,j): x[i,j,k].x for i in customers for j in customers if i != j}
            
            # Find all subtours for this van
            subtours = find_subtours(current_solution, k)
            
            # Add constraints for each subtour
            for subtour in subtours:
                if len(subtour) < len([i for i in customers if y[i,k].x > 0.5]):
                    # Add subtour elimination constraint
                    constraint = model.addConstr(
                        gp.quicksum(x[i,j,k] for i in subtour for j in subtour if i != j) <= len(subtour) - 1
                    )
                    subtour_constraints.append(constraint)
                    subtours_found = True
                    print(f"Added subtour elimination constraint for van {k}: {subtour}")
        
        if not subtours_found:
            break
            
        iteration += 1
    
    # Print results
    print(f"\n=== Optimization Results (with Subtour Elimination) ===")
    print(f"Status: {model.status}")
    print(f"Objective Value (Maximum Weight Difference): {model.objVal:.2f} lbs")
    print(f"Number of Subtour Elimination Constraints Added: {len(subtour_constraints)}")
    
    return model.objVal, len(subtour_constraints)

# Run the model and get results
optimal_value, num_constraints = solve_with_subtour_elimination()
print(f"\nFinal Results:")
print(f"1. Optimal Objective Function Value: {optimal_value:.2f} lbs")
print(f"2. Number of Lazy Constraints Added: {num_constraints}") 