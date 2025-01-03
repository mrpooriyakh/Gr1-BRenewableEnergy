# Actual-solar-iridiance-added
import pandas as pd
from pyomo.environ import *
import matplotlib.pyplot as plt
import json 

# Load Data
# Load time series data
file_path = 'time_series_data.csv'
wind_data_path = 'real-wind-speed.csv'
heating_cooling_data_path = 'adjusted_yearly_heating_cooling_demand.csv'  # Path to the heating/cooling data file

# Load electricity and wind data
data = pd.read_csv(file_path)
wind_data = pd.read_csv(wind_data_path)

# Extract electricity demand
electricity_demand = data['Electricity Demand (kWh)'][:24].values  # Full year (8760 hours)

# Load heating and cooling demand data
heating_cooling_data = pd.read_csv(heating_cooling_data_path)
heating_demand = heating_cooling_data['Heating Demand (kWh)'][:24].values  # Full year
cooling_demand = heating_cooling_data['Cooling Demand (kWh)'][:24].values  # Full year

# Load solar irradiance from cleaned JSON
with open('sorted_cleaned_data.json', 'r') as json_file:
    solar_data = json.load(json_file)
# Extract solar irradiance for each hour
solar_irradiance = [entry['solar_irradiance'] for entry in solar_data]
# Extract Data for the First 24 Hours
electricity_demand = data['Electricity Demand (kWh)'][:24].values
wind_speed = wind_data['wind speed'][:24].values
time_steps = len(electricity_demand)
# Heat Pump Parameters
heat_pump_heating_cop = 3.5  # Heating COP
heat_pump_cooling_cop = 3.0  # Cooling COP
heat_pump_capacity = 10  # Maximum capacity in kW
heat_pump_capex = 5000  # Capital cost ($)
heat_pump_opex_per_kwh = 0.01  # Operational cost per kWh ($/kWh)

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
grid_import_price = 150000000000  # $0.15/kWh import price
grid_export_price = 0.000008  # $0.08/kWh export price

# Battery Parameters - typical residential lithium-ion system
battery_capacity = 13.5  # Tesla Powerwall-like capacity (kWh)
battery_charge_efficiency = 0.95  # 95% charging efficiency
battery_discharge_efficiency = 0.95  # 95% discharging efficiency
battery_max_charge_rate = 5.0  # 5kW maximum charging rate
battery_max_discharge_rate = 5.0  # 5kW maximum discharging rate
battery_initial_soc = 0  # Starting at 50% capacity (kWh)
battery_min_soc = 0.0000001  # 10% minimum state of charge (kWh)
battery_capex = 8500  # $8500 installation cost
battery_cost_per_kwh = 0.0000000000002  # $0.02/kWh operational cost
battery_lifetime_throughput = 40000  # 40MWh lifetime throughput

# Model Definition
model = ConcreteModel()

# Sets
model.T = RangeSet(0, time_steps - 1)

# Variables
# Heat Pump Variables
model.heat_pump_heating_output = Var(model.T, within=NonNegativeReals)
model.heat_pump_cooling_output = Var(model.T, within=NonNegativeReals)
model.heat_pump_mode = Var(model.T, within=Binary)  # Binary mode: heating or cooling

# Grid Variables
model.grid_import = Var(model.T, within=NonNegativeReals)
model.grid_export = Var(model.T, within=NonNegativeReals)
model.grid_import_mode = Var(model.T, within=Binary)

# Renewable Variables
model.solar_pv_output = Var(model.T, within=NonNegativeReals)
model.wind_power_output = Var(model.T, within=NonNegativeReals)
model.Z_solar = Var(within=Binary)
model.Z_wind = Var(within=Binary)
# Heat Pump Binary Variable
model.Z_heat_pump = Var(within=Binary)  # 1 if heat pump is active, 0 otherwise


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
    # Grid Costs
    grid_costs = sum(grid_import_price * model.grid_import[t] for t in model.T)
    grid_revenue = sum(grid_export_price * model.grid_export[t] for t in model.T)
    
    # Renewable Costs
    renewable_costs = sum(pv_cost_per_kwh * model.solar_pv_output[t] +
                          wind_cost_per_kwh * model.wind_power_output[t] 
                          for t in model.T)
    
    # Battery Costs
    battery_costs = sum(battery_cost_per_kwh * (model.battery_charge[t] + 
                                                model.battery_discharge[t])
                        for t in model.T)
    
    # Heat Pump OPEX
    heat_pump_opex = sum((model.heat_pump_heating_output[t] / heat_pump_heating_cop +
                          model.heat_pump_cooling_output[t] / heat_pump_cooling_cop) * heat_pump_opex_per_kwh
                         for t in model.T)
    
    # CAPEX Costs (if applicable)
    capex_costs = (pv_capex * model.Z_solar +
                   wind_capex * model.Z_wind +
                   battery_capex * model.Z_battery +
                   heat_pump_capex * model.Z_heat_pump)
    
    return grid_costs - grid_revenue + renewable_costs + battery_costs + heat_pump_opex + capex_costs

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

# Add a constraint to ensure grid export is only from renewable energy
def grid_export_from_renewable(model, t):
    return model.grid_export[t] <= (model.solar_pv_output[t] + model.wind_power_output[t])

model.grid_export_from_renewable = Constraint(model.T, rule=grid_export_from_renewable)

# Renewable Constraints
def solar_pv_limit(model, t):
    return model.solar_pv_output[t] <= solar_potential[t] * model.Z_solar

def wind_power_limit(model, t):
    return model.wind_power_output[t] <= wind_potential[t] * model.Z_wind

model.solar_pv_limit = Constraint(model.T, rule=solar_pv_limit)
model.wind_power_limit = Constraint(model.T, rule=wind_power_limit)
#heatpump activation check up
def heat_pump_activation_constraint(model, t):
    return (model.heat_pump_heating_output[t] + model.heat_pump_cooling_output[t]) <= \
           heat_pump_capacity * model.Z_heat_pump
model.heat_pump_activation_constraint = Constraint(model.T, rule=heat_pump_activation_constraint)

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
#updated energy balance having added the electricy used by heat pump to electricity demand
def energy_balance_with_heat_pump(model, t):
    heat_pump_electricity = (model.heat_pump_heating_output[t] / heat_pump_heating_cop +
                             model.heat_pump_cooling_output[t] / heat_pump_cooling_cop)
    return (model.grid_import[t] - model.grid_export[t] +
            model.solar_pv_output[t] +
            model.wind_power_output[t] +
            model.battery_discharge[t] ==
            electricity_demand[t] + heat_pump_electricity + model.battery_charge[t])
model.energy_balance_with_heat_pump = Constraint(model.T, rule=energy_balance_with_heat_pump)
def heating_energy_balance(model, t):
    return model.heat_pump_heating_output[t] == heating_demand[t]
model.heating_energy_balance = Constraint(model.T, rule=heating_energy_balance)

def cooling_energy_balance(model, t):
    return model.heat_pump_cooling_output[t] == cooling_demand[t]
model.cooling_energy_balance = Constraint(model.T, rule=cooling_energy_balance)
# Define heating and cooling months
heating_months = {0, 1, 2, 3, 10, 11}  # October to March
cooling_months = {4, 5, 6, 7, 8, 9}   # April to September

def seasonal_operation_constraint(model, t):
    month = (t // 720) % 12  # Approximate month based on hour
    if month in heating_months:
        return model.heat_pump_cooling_output[t] == 0
    elif month in cooling_months:
        return model.heat_pump_heating_output[t] == 0
    else:
        return Constraint.Skip
model.seasonal_operation_constraint = Constraint(model.T, rule=seasonal_operation_constraint)


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
    heat_pump_heating_values = [value(model.heat_pump_heating_output[t]) for t in model.T]
    heat_pump_cooling_values = [value(model.heat_pump_cooling_output[t]) for t in model.T]

    # Calculate Total Electricity Demand (Including Heat Pump)
    total_electricity_demand = [electricity_demand[t] +
                                (heat_pump_heating_values[t] / heat_pump_heating_cop) +
                                (heat_pump_cooling_values[t] / heat_pump_cooling_cop)
                                for t in range(time_steps)]

    # Calculate Key Metrics
    total_demand = sum(electricity_demand)
    total_solar = sum(solar_pv_values)
    total_wind = sum(wind_power_values)
    total_grid_import = sum(grid_import_values)
    total_grid_export = sum(grid_export_values)
    total_battery_charge = sum(battery_charge_values)
    total_battery_discharge = sum(battery_discharge_values)
    total_heating_demand = sum(heating_demand)
    total_cooling_demand = sum(cooling_demand)
    total_heat_pump_electricity = sum((heat_pump_heating_values[t] / heat_pump_heating_cop) +
                                      (heat_pump_cooling_values[t] / heat_pump_cooling_cop)
                                      for t in range(time_steps))

    # Print Results
    print("\n=== System Operation Results ===")
    print(f"Total Electricity Demand (without Heat Pump): {total_demand:.2f} kWh")
    print(f"Total Electricity Demand (with Heat Pump): {sum(total_electricity_demand):.2f} kWh")
    print(f"Heat Pump Electricity Consumption: {total_heat_pump_electricity:.2f} kWh")
    print(f"Solar Generation: {total_solar:.2f} kWh")
    print(f"Wind Generation: {total_wind:.2f} kWh")
    print(f"Grid Import: {total_grid_import:.2f} kWh")
    print(f"Grid Export: {total_grid_export:.2f} kWh")
    print(f"Battery Charge: {total_battery_charge:.2f} kWh")
    print(f"Battery Discharge: {total_battery_discharge:.2f} kWh")
    print(f"Total Heating Demand: {total_heating_demand:.2f} kWh")
    print(f"Total Cooling Demand: {total_cooling_demand:.2f} kWh")

    # === First Plot: Electricity and SOC ===
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Energy Sources and Demand
    plt.subplot(2, 1, 1)
    plt.stackplot(range(time_steps),
                  solar_pv_values,
                  wind_power_values,
                  grid_import_values,
                  battery_discharge_values,
                  labels=['Solar PV', 'Wind', 'Grid Import', 'Battery Discharge'])
    plt.plot(range(time_steps), total_electricity_demand, 'k--', label='Total Electricity Demand')
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

    # === Second Plot: Heating and Cooling ===
    plt.figure(figsize=(15, 10))

    # Plot 3: Heating Demand and Supply
    plt.subplot(2, 1, 1)
    plt.stackplot(range(time_steps),
                  heat_pump_heating_values,
                  labels=['Heat Pump Heating'])
    plt.plot(range(time_steps), heating_demand, 'k--', label='Heating Demand')
    plt.title('Heating Sources and Demand')
    plt.xlabel('Time (hours)')
    plt.ylabel('Power (kW)')
    plt.legend()
    plt.grid(True)

    # Plot 4: Cooling Demand and Supply
    plt.subplot(2, 1, 2)
    plt.stackplot(range(time_steps),
                  heat_pump_cooling_values,
                  labels=['Heat Pump Cooling'])
    plt.plot(range(time_steps), cooling_demand, 'k--', label='Cooling Demand')
    plt.title('Cooling Sources and Demand')
    plt.xlabel('Time (hours)')
    plt.ylabel('Power (kW)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

else:
    print("Failed to find optimal solution")


