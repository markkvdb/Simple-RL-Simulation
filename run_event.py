# Author: Mark van der Broek
# Date: 10-05-2019
# Description: Simple reverse logistic simulation

import pandas as pd

from e_sim.sim_components import Simulator, Dists
from e_sim.utils import experiment_runner_par, experiment_runner


def main():
    # Common settings of the simulation
    sim_time = 1000

    # Settings dict
    settings = {
        "demand_rate": [1],
        "demand_type": [Dists.POISSON],
        "repair_rate": [0.5],
        "repair_type": [Dists.DETERMINISTIC],
        "Q_service": [1, 2, 3, 4, 5, 6, 7, 8],
        "Q_repair": [1, 2, 3, 4, 5, 6, 7, 8],
        "S_depot": [0, 1, 2, 3, 4, 5],
        "S_warehouse": [0, 1, 2, 3, 4, 5],
        "init_stock_depot": [1],
        "init_stock_warehouse": [1],
    }

    # Run all combinations of experiments
    # dfs_agg = experiment_runner(settings, sim_time)
    dfs_agg = experiment_runner_par(settings, sim_time, 6, False)

    df_agg = pd.concat(dfs_agg, ignore_index=True, sort=True)
    df_agg.to_csv("output/experiments/sim_data_agg.csv", index=False)


if __name__ == "__main__":
    main()
