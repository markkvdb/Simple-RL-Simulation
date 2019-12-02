# Author: Mark van der Broek
# Date: 10-05-2019
# Description: Simple reverse logistic simulation

import pandas as pd

from e_sim.sim_components import Simulator
from e_sim.utils import experiment_runner_par

def main():
  # Common settings of the simulation
  sim_time = 10

  # Settings dict
  settings = {}
  settings['demand_rate'] = [1]
  settings['repair_rate'] = [0.5]
  settings['Q_service'] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
  settings['Q_repair'] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
  settings['S_depot'] = [1]
  settings['S_warehouse'] = [2]
  settings['init_stock_depot'] = [2, 4]
  settings['init_stock_warehouse'] = [2, 8]

  # Run all combinations of experiments
  dfs_agg = experiment_runner_par(settings, sim_time, 6, False)

  df_agg = pd.concat(dfs_agg, ignore_index=True, sort=True)
  df_agg.to_csv('output/experiments/sim_data_agg.csv', index=False)

if __name__ == "__main__":
  main()
