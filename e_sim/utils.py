from fnmatch import fnmatch
from os import listdir
from functools import partial

import numpy as np
import pandas as pd
from itertools import product
from tqdm import tqdm
from multiprocessing import Pool

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
  pbar = tqdm(total=len(settings_comb))
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
    pbar.update(1)
  
  pbar.close()
  
  return sim_dfs


def settings_to_fn(settings_experiment: dict, agg: bool = False):
  """Turn experiment settings to file name"""
  fn = ['{}={}'.format(name, value) for name, value in settings_experiment.items()]
  fn = '-'.join(fn)
  if agg: fn = f'output/experiments/agg_{fn}.csv'
  else: fn = f'output/experiments/{fn}.csv'

  return fn


def create_experiment(settings_vals: list,
                      settings_names: list, 
                      sim_time: float,
                      save_sim: bool = True):
  """Compute single experiment and save output to csv."""
  # Create experiment dictionary
  settings = dict(zip(settings_names, settings_vals))

  simulator = Simulator(sim_time, settings)
  simulator.run()
        
  # Add setting columns
  sim_data = simulator.create_output_df()
  for setting, value in settings.items():
    sim_data[setting] = value

  # Get aggregated infO
  df_agg = sim_data.groupby(list(settings.keys())).apply(agg_data)
  df_agg = df_agg.reset_index()

  # Save data as CSV
  fn = settings_to_fn(settings)
  if save_sim: sim_data.to_csv(fn, index=False)
  
  return df_agg


def experiment_runner_par(settings: dict, 
                          sim_time: float, 
                          threads: int = 2,
                          save_sim: bool = True):
  """Runs all model experiments with all different combinations of provided 
  settings values.

  Keyword arguments:
    settings -- Dictionary with all settings
    sim_time -- Number of units of time to simulate
    threads -- Number of jobs to perform in parallel
    save_sim -- Should all output of the simulations be saved

  Returns:
    List with aggregated output DataFrames
  """
  
  # Create all possible combinations of setting values
  settings_names = sorted(settings)
  settings_comb = list(product(*(settings[name] for name in settings_names)))

  p_create_experiment = partial(create_experiment,
                                settings_names=settings_names,
                                sim_time=sim_time,
                                save_sim=save_sim)

  # Run experiment for every combination
  dfs = [None] * len(settings_comb)

  with Pool(threads) as p:
    with tqdm(total=len(settings_comb)) as pbar:
      for i, df in tqdm(enumerate(p.imap_unordered(p_create_experiment, iter(settings_comb)))):
        pbar.update()
        dfs[i] = df

  return dfs

def agg_data(sim_data: pd.DataFrame, keep: bool = False):
  """Compute mean number of shipments (repair and service) and the 
  avg. holding and backorder levels for the simulation."""
  try:
    # Obtain batch size variables
    q_service = sim_data.Q_service.unique()[0]
    q_repair = sim_data.Q_repair.unique()[0]

    # Total simulation time
    sim_time = sim_data.time.iloc[-1]

    # Inventory holding cost
    total_stock = sim_data.init_stock_depot.unique()[0] + sim_data.init_stock_warehouse.unique()[0]

    # Average number of service shipments per unit of time
    total_service_shipments = (np.sum(sim_data.SHIP_SERVICE) / q_service) / sim_time

    # Average number of repair shipments per unit of time
    total_repair_shipments = (np.sum(sim_data.SHIP_REPAIR) / q_repair) / sim_time

    # Avg number of back orders per unit of time
    # First get how long the simulation was in a certain state by differencing
    # the time column.
    back_order_times = np.diff(sim_data.time)

    # The total back-order cost is the sum of the multiplication of the costs
    # at certain times times the time the simulation was in this state.
    avg_back_order_level = np.sum(np.multiply(back_order_times,
                                              sim_data.service_back_orders[:-1]))
    avg_back_order_level /= sim_time

    df_return = pd.DataFrame({
      'avg_stock': total_stock,
      'avg_backorder': avg_back_order_level,
      'service_shipments': total_service_shipments,
      'repair_shipments': total_repair_shipments
    }, index=[0])

  except AttributeError:
    df_return = pd.DataFrame(columns=['avg_stock',
                                      'avg_backorder',
                                      'service_shipments',
                                      'repair_shipments'])
    
  return df_return


def compute_avg_cost(agg_data: pd.DataFrame, costs: dict):
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
    agg_data: dataframe containing aggregated info about the simulation
    costs: dictionary with all cost variables.
  
  Returns:
    Average cost over the entire simulation horizon.
  """
  # Compute costs
  cost_holding = agg_data['avg_stock'] * costs['holding']
  cost_backorder = agg_data['avg_backorder'] * costs['backorder']
  cost_service_ship = agg_data['service_shipments'] * costs['c_service']
  cost_repair_ship = agg_data['repair_shipments'] * costs['c_repair']

  df_results = pd.DataFrame({
    'holding_cost': cost_holding,
    'back_order_cost': cost_backorder,
    'setup_repair_cost': cost_repair_ship,
    'setup_service_cost': cost_service_ship,
    'average_cost': cost_holding + cost_backorder + cost_repair_ship + cost_service_ship
  })

  return df_results

def sensitivity_cost(agg_data: pd.DataFrame, costs: dict, costs_sen: dict):
  """Compute cost for all values of the cost parameters.
  
  The costs dictionary contains all reference cost parameters and costs_sen
  contains the cost parameter values considered for the sensitivity analysis.
  
  Args:
    agg_data: dataframe containing aggregated info about the simulation
    costs: dictionary with all cost variables.
    costs_sen: dictionary with np.arrays of cost parameter values for the
      sensitivity analysis.
  """
  # Create empty dataframe to store results
  df = pd.DataFrame(columns=['cost_par',
                             'holding', 
                             'backorder', 
                             'c_service',
                             'c_repair',
                             'average_cost'])

  # Map cost parameter to simulation component
  cost_map = {
    'holding': 'avg_stock',
    'backorder': 'avg_backorder',
    'c_service': 'service_shipments',
    'c_repair': 'repair_shipments'
  }

  for cost_par_sen, cost_val_sen in costs_sen:
    df_dict = {'cost_par': cost_par_sen}
    total_cost = 0

    for cost_par, cost_val in costs:
      if cost_par == cost_par_sen:
        df_dict[cost_par] = agg_data[cost_map[cost_par]] * cost_val_sen
      else:
        df_dict[cost_par] = agg_data[cost_map[cost_par]] * cost_val
      
      total_cost += df_dict[cost_par]
    
    df_dict['average_cost'] = total_cost
    df = df.append(df.DataFrame(df_dict))
  
  # Rename columns to match with other analysis
  df = df.rename(columns={
    'holding': 'holding_cost',
    'backorder': 'back_order_cost',
    'c_service': 'setup_service_cost',
    'c_repair': 'setup_repair_cost'
  })

  return df
    

def get_sim_fns(path: str):
  """Get all aggregate simulation files of all experiments."""
  return [file for file in listdir(path) if fnmatch(file, 'agg_*')]