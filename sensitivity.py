import sys

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from e_sim.utils import compute_avg_cost, agg_data, sensitivity_cost

def main():
  path_experiment = 'output/experiments/'

  costs = {
    'backorder': 0.4,
    'holding': 0.02,
    'c_service': 1,
    'c_repair': 2
  }

  cost_ranges = {
    'backorder': np.linspace(0, 1),
    'holding': np.linspace(0, 1),
    'c_service': np.linspace(0, 5),
    'c_repair': np.linspace(0, 5)
  }

  settings = {
    'demand_rate': [1],
    'repair_rate': [0.5],
    'Q_service': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Q_repair': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'S_depot': [1],
    'S_warehouse': [2],
    'init_stock_depot': [2, 4],
    'init_stock_warehouse': [2, 8]
  }

  df_agg = pd.read_csv(f'{path_experiment}sim_data_agg.csv', index_col=list(settings.keys()))

  df_sensitivity = df_agg.groupby(list(settings.keys())).apply(sensitivity_cost, costs, cost_ranges)

if __name__ == "__main__":
  main()


