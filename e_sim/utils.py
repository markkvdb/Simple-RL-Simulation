import numpy as np
import pandas as pd
from itertools import product

from .sim_components import Simulator


def experiment_runner(settings, sim_time):
  """Runs all model experiments with all different combinations of provided 
  settings values.

  Keyword arguments:
    settings -- Dictionary with all settings

  Returns:
    A pd.DataFrame with all output info of all experiments with all experiment
    settings as separete columns appended.
  """
  
  # Create all possible combinations of setting values
  settings_names = sorted(settings)
  settings_comb = list(product(*(settings[name] for name in settings_names)))

  sim_dfs = pd.DataFrame()

  # Run experiment for every combination
  for settings_vals in settings_comb:
    settings_experiment = dict(zip(settings_names, settings_vals))

    # Run simulation
    simulator = Simulator(sim_time, settings_experiment)
    simulator.run()
            
    # Save output to master data frame
    sim_data = simulator.create_output_df()
    for setting, value in settings_experiment.items():
      sim_data[setting] = value

    # Add column containing all settings as string
    setting_str = [name + '=' + str(value) for name, value in settings_experiment.items()]
    sim_data['settings'] = ', '.join(setting_str)

    sim_dfs = sim_dfs.append(sim_data)
  
  return(sim_dfs)


def compute_avg_cost(sim_data: pd.DataFrame, costs: dict):
    """Compute the average cost over the entire simulation period.
    
    The cost consists of three parts: inventory holding costs, set-up costs 
    for shipments and back-ordering cost. 
    
    The inventory holding costs are the same since we assume that items can be 
    used indefinetely and the holding cost are identical no matter what states 
    items are in. The set-up cost is the sum of all shipments divided by the 
    simulation time. Finally the back-ordering cost is the back-order cost at
    specific times multiplied by the amount of time the simulation was in this
    state.

    Args:
      event_data: dataframe containing demand, repair and shipments events for
        all observed time periods.
      stock_data: inventory position and levels at sites for all observed time    periods.
    
    Returns:
      Average cost over the entire simulation horizon.
    """

    # Obtain batch size variables
    q_service = sim_data.Q_service.unique()[0]
    q_repair = sim_data.Q_repair.unique()[0]

    # Inventory holding cost
    total_stock = sim_data.init_stock_depot.unique()[0] + sim_data.init_stock_warehouse.unique()[0]
    cost_holding = total_stock * costs['holding']

    # Set-up cost of serviceable items
    cost_setup_service = (np.sum(sim_data.SHIP_SERVICE) / q_service) * costs['c_service']

    # Set-up cost of repairable shipments
    cost_setup_repair = (np.sum(sim_data.SHIP_REPAIR) / q_repair) * costs['c_repair']

    # First get how long the simulation was in a certain state by differencing
    # the time column.
    back_order_times = np.diff(sim_data.time)

    # Back order costs in a specific time are computed as back-order level 
    # multiplied by the back-order cost per item per unit of time.
    back_order_cost = sim_data.service_back_orders * costs['back_order']

    # The total back-order cost is the sum of the multiplication of the costs
    # at certain times times the time the simulation was in this state.
    back_order_cost = np.sum(np.multiply(back_order_times, back_order_cost[:-1]))

    sim_time = sim_data.time.iloc[-1]

    avg_cost = (cost_setup_repair + cost_setup_service + back_order_cost + cost_holding) / sim_time

    return avg_cost

