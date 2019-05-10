# Author: Mark van der Broek
# Date: 10-05-2019
# Description: Simple reverse logistic simulation

import numpy as np
import pandas as pd
from tqdm import tqdm

class Depot(object):
  """Depot class that serves demand and collects repairable units.

  Attributes:
    service_stock: Net stock of servicable units.
    repair_stock: Net stock of repairable units.
    out_server: Server to send units to.
  """

  def __init__(self, batch_size, demand_rate, out_server=None):
    """Initialise Depot class."""
    self._batch_size = batch_size
    self._demand_rate = demand_rate
    self.service_stock = 10
    self.repair_stock = 0
    self.out_server = out_server
    self.sim_time = 0

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
    # Output logger to file
    self.log_data.to_csv('output/depot_output.csv')
    self.log_events.to_csv('output/depot_events.csv')

  def _release_policy(self):
    """Release policy of server."""
    while (self.repair_stock >= self._batch_size):
      self.repair_stock -= self._batch_size
      self.out_server.get_repairables(self._batch_size)
      self.log_events = self.log_events.append({'time': self.sim_time, 'event': 'order'}, ignore_index=True)
    
  def _demand_step(self):
    """Let demand arrive and processes both stocks."""
    
    # Let demand arrive (for now deterministic)
    demand_size = self._demand_rate

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



class Service(object):
  """Service class that repairs repairable units and sends servicable units to
  depot.
  
  Attributes:
    service_stock: Net stock of servicable units.
    repair_stock: Net stock of repairable units.
    out_server: Server to send units to.
  """

  def __init__(self, batch_size, repair_rate, out_server=None):
    """Initialise Service class."""
    self._batch_size = batch_size
    self._repair_rate = repair_rate
    self.service_stock = 0
    self.repair_stock = 10
    self.out_server = out_server
    self.sim_time = 0

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
    # Output logger to file
    self.log_data.to_csv('output/server_output.csv')
    self.log_events.to_csv('output/server_events.csv')

  def get_repairables(self, batch_size):
    """Add incoming repairables to stock."""
    self.repair_stock += batch_size
  
  def _repair_step(self):
    """Handle repair station at time step."""

    # Repair size is minimum of repair stock and repair rate
    repair_size = min(self.repair_stock, self._repair_rate)

    # Update repairable stocks and servicable stocks
    self.repair_stock -= repair_size
    self.service_stock += repair_size
  
  def _release_policy(self):
    """Release policy of server."""
    while (self.service_stock >= self._batch_size):
      self.out_server.take_supply(self._batch_size)
      self.service_stock -= self._batch_size
      self.log_events = self.log_events.append({'time': self.sim_time, 'event': 'order'}, ignore_index=True)

  def _log(self):
    """Log relevant info of simulation."""
    self.log_data = self.log_data.append({'service_stock': self.service_stock, 
                                          'repair_stock': self.repair_stock}, ignore_index=True)


def main():
  # Settings of simulation
  n_time_steps = 1000

  # Settings depot
  depot_batch_size = 5
  depot_demand_rate = 1
  
  # Settings server
  server_batch_size = 1
  server_repair_rate = 2

  # Create simulation objects
  depot = Depot(depot_batch_size, depot_demand_rate)
  server = Service(server_batch_size, server_repair_rate)

  # Link outgoing service stations
  depot.out_server = server
  server.out_server = depot

  for t in tqdm(range(n_time_steps)):
    # Process time step for elements
    depot.step()
    server.step()

  # Save output to file 
  depot.save()
  server.save()

if __name__ == "__main__":
  main()




