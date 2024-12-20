# Required Libraries
import pandas as pd
from pyomo.environ import *
import matplotlib.pyplot as plt

# Load Data
file_path = 'time_series_data.csv'
data = pd.read_csv(file_path)

# Extract Data for the First 24 Hours
electricity_demand = data['Electricity Demand (kWh)'][:24].values
solar_irradiance = data['Solar Irradiance (kW/m2)'][:24].values
time_steps = len(electricity_demand)

# Parameters
pv_efficiency = 0.18  # Efficiency of Solar PV
pv_area = 10  # Area of solar panels in m²
pv_capex = 1000  # Fixed CapEx cost for Solar PV ($)
pv_opex_per_kWh = 0.02  # OpEx cost per kWh produced by Solar PV ($/kWh)
grid_max_import = 10  # Grid import limit (kWh)
grid_price = 0.10  # Grid electricity price ($/kWh)
M = 1000  # Large constant for binary linking constraint

# Step 1: Define the Pyomo Optimization Model
model = ConcreteModel()

# --- Sets ---
model.T = RangeSet(0, time_steps - 1)

# --- Variables ---
model.grid_import = Var(model.T, within=NonNegativeReals)  # Grid import (kWh)
model.solar_pv_output = Var(model.T, within=NonNegativeReals)  # Solar PV output (kWh)
model.Z = Var(within=Binary)  # Binary variable for Solar PV CapEx activation

# --- Parameters ---
demand = {t: electricity_demand[t] for t in model.T}
solar_potential = {t: solar_irradiance[t] * pv_efficiency * pv_area for t in model.T}

# --- Objective Function ---
def total_cost(model):
    return (
        sum(grid_price * model.grid_import[t] + pv_opex_per_kWh * model.solar_pv_output[t] for t in model.T) +
        pv_capex * model.Z
    )

model.cost = Objective(rule=total_cost, sense=minimize)

# --- Constraints ---

# 1. Grid Import Limits
def grid_import_limit(model, t):
    return model.grid_import[t] <= grid_max_import

model.grid_import_limit = Constraint(model.T, rule=grid_import_limit)

# 2. Solar PV Output Limits
def solar_pv_limit(model, t):
    return model.solar_pv_output[t] <= solar_potential[t]

model.solar_pv_limit = Constraint(model.T, rule=solar_pv_limit)

# 3. Energy Balance for Electricity
def energy_balance(model, t):
    return model.grid_import[t] + model.solar_pv_output[t] >= demand[t]

model.energy_balance = Constraint(model.T, rule=energy_balance)

# 4. CapEx Activation Constraint
def solar_capex_activation(model, t):
    return model.solar_pv_output[t] <= M * model.Z

model.capex_activation = Constraint(model.T, rule=solar_capex_activation)

# 5. Solar PV Offline Hours (7 PM to 6 AM)
def solar_pv_offline(model, t):
    hour_of_day = t % 24  # Hour of day from 0 to 23
    if hour_of_day >= 19 or hour_of_day < 6:  # 7 PM to 6 AM
        return model.solar_pv_output[t] == 0
    return Constraint.Skip  # No restriction outside this range

model.solar_pv_offline = Constraint(model.T, rule=solar_pv_offline)

# Step 2: Solve the Model
solver = SolverFactory('glpk')
results = solver.solve(model)

# Check Solver Feasibility
if (results.solver.status != 'ok') or (results.solver.termination_condition != TerminationCondition.optimal):
    print("Solver failed to find an optimal solution. Please check your constraints and inputs.")
    exit()

# Step 3: Extract Results
grid_import_values = [model.grid_import[t]() if model.grid_import[t]() is not None else 0 for t in model.T]
solar_pv_values = [model.solar_pv_output[t]() if model.solar_pv_output[t]() is not None else 0 for t in model.T]
capex_incurred = model.Z() if model.Z() is not None else 0

# Step 4: Calculations for Costs and Participation
total_grid_cost = sum(grid_price * grid_import_values[t] for t in range(time_steps))
total_solar_cost = sum(pv_opex_per_kWh * solar_pv_values[t] for t in range(time_steps))
total_system_cost = total_grid_cost + total_solar_cost + (pv_capex if capex_incurred > 0 else 0)
total_demand = sum(electricity_demand)
total_solar = sum(solar_pv_values)
total_grid = sum(grid_import_values)

solar_percentage = (total_solar / total_demand) * 100
grid_percentage = (total_grid / total_demand) * 100

# Print Results
print(f"--- Results Summary ---")
print(f"Total Electricity Demand: {total_demand:.2f} kWh")
print(f"Total Solar PV Contribution: {total_solar:.2f} kWh ({solar_percentage:.2f}%)")
print(f"Total Grid Import Contribution: {total_grid:.2f} kWh ({grid_percentage:.2f}%)")
print(f"Total Grid Electricity Cost: ${total_grid_cost:.2f}")
print(f"Total Solar PV OpEx Cost: ${total_solar_cost:.2f}")
if capex_incurred > 0:
    print(f"Solar PV CapEx Cost: ${pv_capex:.2f}")
print(f"Total System Cost: ${total_system_cost:.2f}")

# Step 5: Visualization
plt.figure(figsize=(12, 6))

# Plot Demand
plt.plot(range(time_steps), electricity_demand, label="Electricity Demand (kWh)", linestyle='--', color='black')

# Plot Solar PV Contribution
plt.fill_between(range(time_steps), 0, solar_pv_values, label="Solar PV Contribution (kWh)", color='orange', alpha=0.7)

# Plot Grid Import Contribution
plt.fill_between(range(time_steps), solar_pv_values, 
                 [solar_pv_values[t] + grid_import_values[t] for t in range(time_steps)],
                 label="Grid Import Contribution (kWh)", color='blue', alpha=0.5)

# Final plot setup
plt.title("Electricity Demand Met by Solar PV and Grid Import (First 24 Hours)")
plt.xlabel("Time Step (Hour)")
plt.ylabel("Energy (kWh)")
plt.legend(loc="upper left")
plt.grid(True)
plt.show()
