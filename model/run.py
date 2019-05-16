# Author: Mark van der Broek
# Date: 10-05-2019
# Description: Simple reverse logistic simulation

import numpy as np
import pandas as pd
from tqdm import tqdm


class Simulator(object):
  """The Simulator class initiates and updates all entities in the simulation 

  Attributes:
    sim_entities: Array of all entities of the simulation (in order)
    delta_time: Fraction of one unit of time for step size
    sim_time: Total units of time for simulation
  """

  def __init__(self, sim_entities, delta_time, sim_time):
    """Initialise Simulator class"""
    self.sim_entities = sim_entities
    self.delta_time = delta_time
    self.sim_time = sim_time
    self.n_steps = int(sim_time / delta_time)
  
  def initialise(self):
    """Set parameters of the simulator to all entities"""
    # For now, this is done manually by updating all entities
    # [0] is customer, [1] is depot
    self.sim_entities[0].demand_rate *= self.delta_time
    self.sim_entities[1].repair_rate *= self.delta_time

  def _step(self):
    """Simulate one time step for all entities (in order)"""
    for entity in self.sim_entities:
      entity.step()
  
  def run(self):
    """Run simulation"""
    for t in tqdm(range(self.n_steps)):
      self._step()
  
  def save(self):
    """Save output of the simulation"""
    for entity in self.sim_entities:
      entity.save()

  def output_df(self):
    """After the simulation is complete, create a neat data frame"""
    data_customer = self.sim_entities[0].log_data
    data_depot = self.sim_entities[1].log_data

    # Combine datasets
    data_customer.columns = 'customer_' + data_customer.columns
    data_depot.columns = "depot_" + data_depot.columns
    all_data = pd.concat([data_customer, data_depot], axis=1)

    # Add time column
    all_data['time'] = np.linspace(start=0, stop=self.sim_time, num=self.n_steps)

    return(all_data)


class Customer(object):
  """Customer class that uses units and sends repairs.

  Attributes:
    service_stock: Net stock of servicable units.
    repair_stock: Net stock of repairable units.
    out_server: Server to send units to.
  """

  def __init__(self, batch_size, demand_rate, out_server=None):
    """Initialise Depot class."""
    self.batch_size = batch_size
    self.demand_rate = demand_rate
    self.service_stock = batch_size
    self.repair_stock = 0
    self.out_server = out_server
    self.sim_time = 0
    self.batch_model = batch_size > 0

    self.log_data = pd.DataFrame(columns=['service_stock', 'repair_stock'])
    self.log_events = pd.DataFrame(columns=['time', 'event'])

  
  def step(self):
    """Simulate one step of simulation for depot."""
    
    # Demand arrives 
    self._demand_step()

    # Release policy
    self._release_policy()

    # Log step
    self._log()
    self.sim_time += 1
  
  def save(self):
    """Output log file to CSV"""
    self.log_data.to_csv('output/customer_output.csv')
    self.log_events.to_csv('output/customer_events.csv')

  def _release_policy(self):
    """Release policy of server."""
    if self.batch_model:
      while (self.repair_stock >= self.batch_size):
        self.repair_stock -= self.batch_size
        self.out_server.get_repairables(self.batch_size)
        self.log_events = self.log_events.append({'time': self.sim_time, 'event': 'order'}, ignore_index=True)
    else:
      self.out_server.get_repairables(self.repair_stock)
      self.repair_stock = 0
    
  def _demand_step(self):
    """Let demand arrive and processes both stocks."""
    
    # Let demand arrive (for now deterministic)
    demand_size = self.demand_rate

    # Repairable stock increases with demand and service stock lowers
    self.repair_stock += demand_size
    self.service_stock -= demand_size

  def take_supply(self, batch_size):
    """Process incoming servicables."""
    self.service_stock += batch_size

  def _log(self):
    """Log relevant info of simulation."""
    self.log_data = self.log_data.append({'service_stock': self.service_stock, 
                                          'repair_stock': self.repair_stock}, ignore_index=True)


class Depot(object):
  """Depot class that repairs repairable units and sends servicable units to
  customer.
  
  Attributes:
    service_stock: Net stock of servicable units.
    repair_stock: Net stock of repairable units.
    out_server: Server to send units to.
  """

  def __init__(self, batch_size, repair_rate, out_server=None):
    """Initialise Service class."""
    self.batch_size = batch_size
    self.repair_rate = repair_rate
    self.service_stock = batch_size
    self.repair_stock = 0
    self.out_server = out_server
    self.sim_time = 0
    self.batch_model = batch_size > 0

    self.log_data = pd.DataFrame(columns=['service_stock', 'repair_stock'])
    self.log_events = pd.DataFrame(columns=['time', 'event'])

  def step(self):
    """Simulate one step of simulation for Service server."""
    
    # Check repair step
    self._repair_step()

    # Update inventories and check release 
    self._release_policy()

    # Log info
    self._log()
    self.sim_time += 1

  def save(self):
    """Output logger df to file"""
    self.log_data.to_csv('output/depot_output.csv')
    self.log_events.to_csv('output/depot_events.csv')

  def get_repairables(self, batch_size):
    """Add incoming repairables to stock."""
    self.repair_stock += batch_size
  
  def _repair_step(self):
    """Handle repair station at time step."""

    # Repair size is minimum of repair stock and repair rate
    repair_size = min(self.repair_stock, self.repair_rate)

    # Update repairable stocks and servicable stocks
    self.repair_stock -= repair_size
    self.service_stock += repair_size
  
  def _release_policy(self):
    """Release policy of server."""
    if self.batch_model:
      while (self.service_stock >= self.batch_size):
        self.out_server.take_supply(self.batch_size)
        self.service_stock -= self.batch_size
        self.log_events = self.log_events.append({'time': self.sim_time, 'event': 'order'}, ignore_index=True)
    else:
      self.out_server.take_supply(self.service_stock)
      self.service_stock = 0

  def _log(self):
    """Log relevant info of simulation."""
    self.log_data = self.log_data.append({'service_stock': self.service_stock, 
                                          'repair_stock': self.repair_stock}, ignore_index=True)


def main():
  # Settings of simulation
  sim_time = 20
  time_delta = 0.01

  # Settings depot
  customer_batch_size = 5
  customer_demand_rate = 1

  # Settings server
  depot_batch_size = 1
  depot_repair_rate = 2

  # Create simulation objects
  customer = Customer(customer_batch_size, customer_demand_rate)
  depot = Depot(depot_batch_size, depot_repair_rate)

  # Link outgoing service stations
  customer.out_server = depot
  depot.out_server = customer

  # Create and run simulator
  simulator = Simulator([customer, depot], time_delta, sim_time)
  simulator.initialise()
  simulator.run()
  simulator.save()

if __name__ == "__main__":
  main()
