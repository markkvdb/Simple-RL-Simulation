# Author: Mark van der Broek
# Date: 10-05-2019
# Description: Simple reverse logistic simulation

from e_sim.sim_components import Simulator
from e_sim.utils import experiment_runner

def main():
  # Common settings of the simulation
  sim_time = 100

  # Settings dict
  settings = {}
  settings['demand_rate'] = [1]
  settings['repair_rate'] = [0.5]
  settings['Q_service'] = [1,2]
  settings['Q_repair'] = [2, 4]
  settings['S_depot'] = [1]
  settings['S_warehouse'] = [2]
  settings['init_stock_depot'] = [2]
  settings['init_stock_warehouse'] = [2, 8]
  settings['b_cost'] = 0.4
  settings['h_cost'] = 0.02

  # Run all combinations of experiments
  sim_dfs = experiment_runner(settings, sim_time)

if __name__ == "__main__":
  main()
