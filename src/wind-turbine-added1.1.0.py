import pandas as pd
from pyomo.environ import *
import matplotlib.pyplot as plt

# Load Electricity Demand and Solar Irradiance Data
file_path = 'time_series_data.csv'
data = pd.read_csv(file_path)

# Load Wind Speed Data
wind_data_path = 'real-wind-speed.csv'  # File containing wind speed data
wind_data = pd.read_csv(wind_data_path)
wind_speed = wind_data['wind speed'].values  # Extract wind speed array

# Extract Other Data for the First 24 Hours
electricity_demand = data['Electricity Demand (kWh)'][:24].values
solar_irradiance = data['Solar Irradiance (kW/m2)'][:24].values
time_steps = len(electricity_demand)

# Parameters
pv_efficiency = 0.18  # Efficiency of Solar PV
pv_area = 10  # Area of solar panels in m²
pv_cost_per_kWh = 0.20  # Levelized cost of solar PV ($/kWh)
pv_capex = 1000  # Fixed CapEx cost for Solar PV ($)
grid_price = 0.10  # Grid electricity price ($/kWh)
grid_max_import = 10  # Grid import limit (kWh)

# Wind Turbine Parameters
wind_turbine_rated_power = 5  # Rated power (kW)
wind_cut_in_speed = 3  # Cut-in wind speed (m/s)
wind_rated_speed = 12  # Rated wind speed (m/s)
wind_cut_off_speed = 25  # Cut-off wind speed (m/s)
wind_cost_per_kWh = 0.15  # OpEx cost per kWh produced by wind turbine ($/kWh)
wind_capex = 2000  # Fixed CapEx cost for Wind Turbine ($)
M = 1000  # Large constant for binary linking constraint

# Step 1: Define the Pyomo Optimization Model
model = ConcreteModel()

# --- Sets ---
model.T = RangeSet(0, time_steps - 1)

# --- Variables ---
model.grid_import = Var(model.T, within=NonNegativeReals)  # Grid import (kWh)
model.solar_pv_output = Var(model.T, within=NonNegativeReals)  # Solar PV output (kWh)
model.wind_power_output = Var(model.T, within=NonNegativeReals)  # Wind power output (kWh)
model.Z_solar = Var(within=Binary)  # Binary variable for Solar PV CapEx activation
model.Z_wind = Var(within=Binary)  # Binary variable for Wind Turbine CapEx activation

# --- Parameters ---
demand = {t: electricity_demand[t] for t in model.T}
solar_potential = {t: solar_irradiance[t] * pv_efficiency * pv_area for t in model.T}

# Wind Power Calculation Function
def wind_power_availability(speed):
    if speed < wind_cut_in_speed or speed > wind_cut_off_speed:
        return 0
    elif wind_cut_in_speed <= speed <= wind_rated_speed:
        return wind_turbine_rated_power * ((speed - wind_cut_in_speed) ** 3) / ((wind_rated_speed - wind_cut_in_speed) ** 3)
    elif wind_rated_speed <= speed <= wind_cut_off_speed:
        return wind_turbine_rated_power
    return 0

# Calculate Wind Potential for Each Time Step
wind_potential = {t: wind_power_availability(wind_speed[t]) for t in model.T}

# --- Objective Function ---
def total_cost(model):
    return (
        sum(grid_price * model.grid_import[t] + pv_cost_per_kWh * model.solar_pv_output[t] +
            wind_cost_per_kWh * model.wind_power_output[t] for t in model.T) +
        pv_capex * model.Z_solar +
        wind_capex * model.Z_wind
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

# 3. Wind Power Output Limits
def wind_power_limit(model, t):
    return model.wind_power_output[t] <= wind_potential[t]

model.wind_power_limit = Constraint(model.T, rule=wind_power_limit)

# 4. Energy Balance for Electricity
def energy_balance(model, t):
    return model.grid_import[t] + model.solar_pv_output[t] + model.wind_power_output[t] >= demand[t]

model.energy_balance = Constraint(model.T, rule=energy_balance)

# 5. CapEx Activation Constraints
def solar_capex_activation(model, t):
    return model.solar_pv_output[t] <= M * model.Z_solar

model.solar_capex_activation = Constraint(model.T, rule=solar_capex_activation)

def wind_capex_activation(model, t):
    return model.wind_power_output[t] <= M * model.Z_wind

model.wind_capex_activation = Constraint(model.T, rule=wind_capex_activation)

# Step 2: Solve the Model
solver = SolverFactory('glpk')
results = solver.solve(model)

# Check Solver Feasibility
if (results.solver.status != 'ok') or (results.solver.termination_condition != TerminationCondition.optimal):
    print("Solver failed to find an optimal solution. Please check your constraints and inputs.")
    exit()

# Step 3: Extract Results
grid_import_values = [model.grid_import[t]() for t in model.T]
solar_pv_values = [model.solar_pv_output[t]() for t in model.T]
wind_power_values = [model.wind_power_output[t]() for t in model.T]
capex_solar_incurred = model.Z_solar() if model.Z_solar() is not None else 0
capex_wind_incurred = model.Z_wind() if model.Z_wind() is not None else 0

# Step 4: Calculations for Costs and Participation
total_grid_cost = sum(grid_price * grid_import_values[t] for t in range(time_steps))
total_solar_cost = sum(pv_cost_per_kWh * solar_pv_values[t] for t in range(time_steps))
total_wind_cost = sum(wind_cost_per_kWh * wind_power_values[t] for t in range(time_steps))
total_system_cost = (
    total_grid_cost +
    total_solar_cost + (pv_capex if capex_solar_incurred > 0 else 0) +
    total_wind_cost + (wind_capex if capex_wind_incurred > 0 else 0)
)
total_demand = sum(electricity_demand)
total_solar = sum(solar_pv_values)
total_wind = sum(wind_power_values)
total_grid = sum(grid_import_values)

solar_percentage = (total_solar / total_demand) * 100
wind_percentage = (total_wind / total_demand) * 100
grid_percentage = (total_grid / total_demand) * 100

# Print Results
print(f"--- Results Summary ---")
print(f"Total Electricity Demand: {total_demand:.2f} kWh")
print(f"Total Solar PV Contribution: {total_solar:.2f} kWh ({solar_percentage:.2f}%)")
print(f"Total Wind Power Contribution: {total_wind:.2f} kWh ({wind_percentage:.2f}%)")
print(f"Total Grid Import Contribution: {total_grid:.2f} kWh ({grid_percentage:.2f}%)")
print(f"Total Grid Electricity Cost: ${total_grid_cost:.2f}")
print(f"Total Solar PV Cost (OpEx + CapEx): ${total_solar_cost + (pv_capex if capex_solar_incurred > 0 else 0):.2f}")
print(f"Total Wind Power Cost (OpEx + CapEx): ${total_wind_cost + (wind_capex if capex_wind_incurred > 0 else 0):.2f}")
print(f"Total System Cost: ${total_system_cost:.2f}")

# Step 5: Visualization
plt.figure(figsize=(12, 6))

# Plot Demand
plt.plot(range(time_steps), electricity_demand, label="Electricity Demand (kWh)", linestyle='--', color='black')

# Plot Solar PV Contribution
plt.fill_between(range(time_steps), 0, solar_pv_values, label="Solar PV Contribution (kWh)", color='orange', alpha=0.7)

# Plot Wind Power Contribution
plt.fill_between(range(time_steps), [solar_pv_values[t] for t in range(time_steps)],
                 [solar_pv_values[t] + wind_power_values[t] for t in range(time_steps)],
                 label="Wind Power Contribution (kWh)", color='green', alpha=0.5)

# Plot Grid Import Contribution
plt.fill_between(range(time_steps), 
                 [solar_pv_values[t] + wind_power_values[t] for t in range(time_steps)],
                 [solar_pv_values[t] + wind_power_values[t] + grid_import_values[t] for t in range(time_steps)],
                 label="Grid Import Contribution (kWh)", color='blue', alpha=0.5)

# Final plot setup
plt.title("Electricity Demand Met by Solar PV, Wind, and Grid Import (First 24 Hours)")
plt.xlabel("Time Step (Hour)")
plt.ylabel("Energy (kWh)")
plt.legend(loc="upper left")
plt.grid(True)
plt.show()
