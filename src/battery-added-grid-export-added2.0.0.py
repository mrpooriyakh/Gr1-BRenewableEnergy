#may
import pandas as pd
from pyomo.environ import *
import matplotlib.pyplot as plt

# Load Data
# Load time series data
file_path = 'time_series_data.csv'
wind_data_path = 'real-wind-speed.csv'

# Read required data files
data = pd.read_csv(file_path)
wind_data = pd.read_csv(wind_data_path)

# Extract Data for the First 24 Hours
electricity_demand = data['Electricity Demand (kWh)'][:24].values
solar_irradiance = data['Solar Irradiance (kW/m2)'][:24].values
wind_speed = wind_data['wind speed'][:24].values
time_steps = len(electricity_demand)

# Solar PV Parameters - typical residential system
pv_efficiency = 0.20  # 20% efficient panels (modern panels)
pv_area = 30  # 30 m² (approximately 5kW system)
pv_cost_per_kwh = 0.03  # $0.03/kWh operational cost
pv_capex = 8000  # $8000 for 5kW system

# Wind Turbine Parameters - small residential turbine
wind_turbine_rated_power = 3  # 3kW rated power
wind_cut_in_speed = 3  # 3 m/s cut-in speed
wind_rated_speed = 12  # 12 m/s rated speed
wind_cut_off_speed = 25  # 25 m/s cut-off speed
wind_cost_per_kwh = 0.05  # $0.05/kWh operational cost
wind_capex = 12000  # $12000 for 3kW turbine

# Grid Parameters - typical residential connection
grid_max_import = 10  # 10kW maximum import capacity
grid_max_export = 8  # 8kW maximum export capacity
grid_import_price = 0.15  # $0.15/kWh import price
grid_export_price = 0.08  # $0.08/kWh export price

# Battery Parameters - typical residential lithium-ion system
battery_capacity = 13.5  # Tesla Powerwall-like capacity (kWh)
battery_charge_efficiency = 0.95  # 95% charging efficiency
battery_discharge_efficiency = 0.95  # 95% discharging efficiency
battery_max_charge_rate = 5.0  # 5kW maximum charging rate
battery_max_discharge_rate = 5.0  # 5kW maximum discharging rate
battery_initial_soc = 0  # Starting at 50% capacity (kWh)
battery_min_soc = 10  # 10% minimum state of charge (kWh)
battery_capex = 8500  # $8500 installation cost
battery_cost_per_kwh = 0.02  # $0.02/kWh operational cost
battery_lifetime_throughput = 40000  # 40MWh lifetime throughput

# Model Definition
model = ConcreteModel()

# Sets
model.T = RangeSet(0, time_steps - 1)

# Variables
# Grid Variables
model.grid_import = Var(model.T, within=NonNegativeReals)
model.grid_export = Var(model.T, within=NonNegativeReals)
model.grid_import_mode = Var(model.T, within=Binary)

# Renewable Variables
model.solar_pv_output = Var(model.T, within=NonNegativeReals)
model.wind_power_output = Var(model.T, within=NonNegativeReals)
model.Z_solar = Var(within=Binary)
model.Z_wind = Var(within=Binary)

# Battery Variables
model.battery_soc = Var(model.T, within=NonNegativeReals)
model.battery_charge = Var(model.T, within=NonNegativeReals)
model.battery_discharge = Var(model.T, within=NonNegativeReals)
model.battery_charge_mode = Var(model.T, within=Binary)
model.Z_battery = Var(within=Binary)

# Parameters
demand = {t: electricity_demand[t] for t in model.T}
solar_potential = {t: solar_irradiance[t] * pv_efficiency * pv_area for t in model.T}

# Wind Power Calculation Function
def wind_power_availability(speed):
    if speed < wind_cut_in_speed or speed > wind_cut_off_speed:
        return 0
    elif wind_cut_in_speed <= speed <= wind_rated_speed:
        return wind_turbine_rated_power * ((speed - wind_cut_in_speed) ** 3) / \
               ((wind_rated_speed - wind_cut_in_speed) ** 3)
    elif wind_rated_speed <= speed <= wind_cut_off_speed:
        return wind_turbine_rated_power
    return 0

wind_potential = {t: wind_power_availability(wind_speed[t]) for t in model.T}

# Objective Function
def total_cost(model):
    grid_costs = sum(grid_import_price * model.grid_import[t] for t in model.T)
    grid_revenue = sum(grid_export_price * model.grid_export[t] for t in model.T)
    
    renewable_costs = sum(pv_cost_per_kwh * model.solar_pv_output[t] +
                         wind_cost_per_kwh * model.wind_power_output[t] 
                         for t in model.T)
    
    battery_costs = sum(battery_cost_per_kwh * (model.battery_charge[t] + 
                                               model.battery_discharge[t])
                       for t in model.T)
    
    capex_costs = (pv_capex * model.Z_solar +
                   wind_capex * model.Z_wind +
                   battery_capex * model.Z_battery)
    
    return grid_costs - grid_revenue + renewable_costs + battery_costs + capex_costs

model.cost = Objective(rule=total_cost, sense=minimize)

# Constraints

# Grid Constraints
def grid_import_limit(model, t):
    return model.grid_import[t] <= grid_max_import

def grid_export_limit(model, t):
    return model.grid_export[t] <= grid_max_export

def no_simultaneous_import_export(model, t):
    return model.grid_import[t] + model.grid_export[t] <= \
           max(grid_max_import, grid_max_export)

model.grid_import_limit = Constraint(model.T, rule=grid_import_limit)
model.grid_export_limit = Constraint(model.T, rule=grid_export_limit)
model.no_simultaneous_import_export = Constraint(model.T, 
                                               rule=no_simultaneous_import_export)

# Renewable Constraints
# First, remove any existing solar constraints if they exist
if hasattr(model, 'solar_pv_limit'):
    model.del_component('solar_pv_limit')
if hasattr(model, 'solar_offline'):
    model.del_component('solar_offline')

def solar_pv_limit(model, t):
    return model.solar_pv_output[t] <= solar_potential[t] * model.Z_solar

def solar_offline_hours(model, t):
    hour_of_day = t % 24  # Convert timestep to hour of day
    if hour_of_day >= 19 or hour_of_day < 6:  # Between 7 PM and 6 AM
        return model.solar_pv_output[t] == 0
    return Constraint.Skip  # No restriction during daylight hours

def wind_power_limit(model, t):
    return model.wind_power_output[t] <= wind_potential[t] * model.Z_wind

# Add constraints to model
model.solar_pv_limit = Constraint(model.T, rule=solar_pv_limit)
model.solar_offline = Constraint(model.T, rule=solar_offline_hours)
model.wind_power_limit = Constraint(model.T, rule=wind_power_limit)

# Battery Constraints
def battery_soc_evolution(model, t):
    if t == 0:
        return model.battery_soc[t] == battery_initial_soc + \
               (battery_charge_efficiency * model.battery_charge[t]) - \
               (model.battery_discharge[t] / battery_discharge_efficiency)
    return model.battery_soc[t] == model.battery_soc[t-1] + \
           (battery_charge_efficiency * model.battery_charge[t]) - \
           (model.battery_discharge[t] / battery_discharge_efficiency)

def battery_min_soc_limit(model, t):
    return model.battery_soc[t] >= battery_min_soc * model.Z_battery

def battery_max_soc_limit(model, t):
    return model.battery_soc[t] <= battery_capacity * model.Z_battery

def battery_charge_limit(model, t):
    return model.battery_charge[t] <= battery_max_charge_rate * \
           model.battery_charge_mode[t]

def battery_discharge_limit(model, t):
    return model.battery_discharge[t] <= battery_max_discharge_rate * \
           (1 - model.battery_charge_mode[t])

def battery_throughput_limit(model):
    return sum(model.battery_charge[t] + model.battery_discharge[t] 
              for t in model.T) <= battery_lifetime_throughput

model.battery_soc_evolution = Constraint(model.T, rule=battery_soc_evolution)
model.battery_min_soc = Constraint(model.T, rule=battery_min_soc_limit)
model.battery_max_soc = Constraint(model.T, rule=battery_max_soc_limit)
model.battery_charge_limit = Constraint(model.T, rule=battery_charge_limit)
model.battery_discharge_limit = Constraint(model.T, rule=battery_discharge_limit)
model.battery_throughput_limit = Constraint(rule=battery_throughput_limit)

# Energy Balance Constraint
def energy_balance(model, t):
    return (model.grid_import[t] - model.grid_export[t] +
            model.solar_pv_output[t] + 
            model.wind_power_output[t] + 
            model.battery_discharge[t] == 
            demand[t] + 
            model.battery_charge[t])

model.energy_balance = Constraint(model.T, rule=energy_balance)

# Solve the Model
solver = SolverFactory('glpk')
results = solver.solve(model)

# Check Solution Status
if (results.solver.status == SolverStatus.ok) and \
   (results.solver.termination_condition == TerminationCondition.optimal):
    
    # Extract Results
    grid_import_values = [value(model.grid_import[t]) for t in model.T]
    grid_export_values = [value(model.grid_export[t]) for t in model.T]
    solar_pv_values = [value(model.solar_pv_output[t]) for t in model.T]
    wind_power_values = [value(model.wind_power_output[t]) for t in model.T]
    battery_soc_values = [value(model.battery_soc[t]) for t in model.T]
    battery_charge_values = [value(model.battery_charge[t]) for t in model.T]
    battery_discharge_values = [value(model.battery_discharge[t]) for t in model.T]

    # Calculate Key Metrics
    total_demand = sum(electricity_demand)
    total_solar = sum(solar_pv_values)
    total_wind = sum(wind_power_values)
    total_grid_import = sum(grid_import_values)
    total_grid_export = sum(grid_export_values)
    total_battery_charge = sum(battery_charge_values)
    total_battery_discharge = sum(battery_discharge_values)

    # Print Results
    print("\n=== System Operation Results ===")
    print(f"Total Demand: {total_demand:.2f} kWh")
    print(f"Solar Generation: {total_solar:.2f} kWh")
    print(f"Wind Generation: {total_wind:.2f} kWh")
    print(f"Grid Import: {total_grid_import:.2f} kWh")
    print(f"Grid Export: {total_grid_export:.2f} kWh")
    print(f"Battery Charge: {total_battery_charge:.2f} kWh")
    print(f"Battery Discharge: {total_battery_discharge:.2f} kWh")

    # Visualization
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Energy Sources and Demand
    plt.subplot(2, 1, 1)
    plt.stackplot(range(time_steps),
                 solar_pv_values,
                 wind_power_values,
                 grid_import_values,
                 battery_discharge_values,
                 labels=['Solar PV', 'Wind', 'Grid Import', 'Battery Discharge'])
    plt.plot(range(time_steps), electricity_demand, 'k--', label='Demand')
    plt.title('Energy Sources and Demand')
    plt.xlabel('Time (hours)')
    plt.ylabel('Power (kW)')
    plt.legend()
    plt.grid(True)

    # Plot 2: Battery Operation
    plt.subplot(2, 1, 2)
    plt.plot(range(time_steps), battery_soc_values, 'b-', label='State of Charge')
    plt.fill_between(range(time_steps), 0, battery_charge_values, 
                     alpha=0.3, color='g', label='Charging')
    plt.fill_between(range(time_steps), 0, battery_discharge_values, 
                     alpha=0.3, color='r', label='Discharging')
    plt.title('Battery Operation')
    plt.xlabel('Time (hours)')
    plt.ylabel('Energy (kWh)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

else:
    print("Failed to find optimal solution")