import cProfile

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
        "Q_service": [2],
        "Q_repair": [4],
        "S_depot": [2],
        "S_warehouse": [4],
        "init_stock_depot": [1],
        "init_stock_warehouse": [1],
    }

    # Run all combinations of experiments
    dfs_agg = experiment_runner(settings, sim_time)


if __name__ == "__main__":
    cProfile.run("main()", "profile_stats")

