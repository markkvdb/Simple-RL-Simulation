# Author: Mark van der Broek
# Date: 10-05-2019
# Description: Simple reverse logistic simulation

from e_sim.sim_components import Simulator

def main():
  # Common settings of the simulation
  sim_time = 1000

  # Settings dict
  settings = {}
  settings['demand_rate'] = 1
  settings['repair_rate'] = 0.5
  settings['Q_service'] = 2
  settings['Q_repair'] = 4
  settings['S_depot'] = 2
  settings['S_warehouse'] = 2
  settings['init_stock_depot'] = 4
  settings['init_stock_warehouse'] = 2

  # Create and run simulator
  simulator = Simulator(sim_time, settings)
  simulator.run()

  # Save
  simulator.save()

  # Get data
  stock_info = simulator.create_output_df()
  stock_info.to_csv('output/neat_df.csv')

if __name__ == "__main__":
  main()
