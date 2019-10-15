# Author: Mark van der Broek
# Date: 10-05-2019
# Description: Simple reverse logistic simulation

from sim_components import Warehouse, Depot, Simulator

def main():
  # Settings of simulation
  sim_time = 20
  time_delta = 0.01

  # Create and run simulator
  simulator = Simulator(time_delta, sim_time)
  simulator.run()

  # Save
  simulator.save()

if __name__ == "__main__":
  main()
