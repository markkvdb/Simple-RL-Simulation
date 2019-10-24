# Author: Mark van der Broek
# Date: 05-08-2019
# Description: Simple reverse logistic simulation

from sys import exit
import numpy as np
import pandas as pd
from tqdm import tqdm


class Simulator(object):
  """The Simulator class initiates and updates all entities in the simulation

  Attributes:
    sim_entities: Array of all entities of the simulation (in order)
    delta_time: Fraction of one unit of time for step size
    sim_time: Total units of time for simulation

  Methods:
    initialise: 
  """

  def __init__(self, delta_time, sim_time, settings):
    """Initialise Simulator class"""

    # Time of simulation attributes
    self.delta_time = delta_time
    self.sim_time = sim_time
    self.n_steps = int(sim_time / delta_time)

    # Inventory model
    self.model = InventoryModel(Q_service = settings['Q_service'], 
                                Q_repair = settings['Q_repair'],
                                S_depot = settings['S_depot'],
                                S_warehouse = settings['S_warehouse'],
                                demand_rate = settings['demand_rate'] * delta_time,
                                repair_rate = settings['repair_rate'] * delta_time,
                                init_stock_depot = settings['init_stock_depot'],
                                init_stock_warehouse = settings['init_stock_warehouse']
    )

  def run(self):
    """Run simulation"""
    for t in tqdm(range(self.n_steps)):
      self.model.sim_step()


  def create_output_df(self):
    return self.model.create_output_df()


  def save(self):
    """After the simulation is complete, create a neat data frame"""
    self.model.save()


class InventoryModel(object):
  """This object holds all the entities of the inventory model and its 
  interactions.
  
  Attributes:
    entities: list of entities used in the simulation
  """

  def __init__(self, Q_service, Q_repair, S_depot, S_warehouse, demand_rate, 
               repair_rate, init_stock_depot, init_stock_warehouse):
    """Initialise InventoryModel class."""
    # Entities of model
    self.depot = Depot(demand_rate, init_stock_depot)
    self.warehouse = Warehouse(repair_rate, init_stock_warehouse)

    # Policy settings
    self.q_service = Q_service
    self.q_repair = Q_repair
    self.s_depot = S_depot
    self.s_warehouse = S_warehouse

    # Costs
    self.c_service = 1
    self.c_repair = 3
    self.h_service = 0.02
    self.b_service = 0.4
    self.h_repair = 0.01

    # Logging info
    self.time = 0
    self.event_data = pd.DataFrame(columns=['time', 'event', 'quantity'])


  def policy_upstream(self):
    """Order policy of the depot. Once inventory position drops below
    S, we initiate an order of size Q."""

    # Place orders as long as inventory position is below S
    while (self.depot.get_service_inventory_position() < self.s_depot):
      self.depot.place_order(self.q_service)

    # Send orders if they are available
    transport_order_size = self.warehouse.get_order(self.depot.service_stock_order, self.q_service)

    if transport_order_size > 0:
      self.depot.get_serviceables(transport_order_size)

      # Log order
      self.event_data = self.event_data.append({
        'time': self.time,
        'event': 'service_order',
        'quantity': int(transport_order_size / self.q_service)
        }, ignore_index=True)


  def policy_downstream(self):
    """Send repairable units to warehouse once we collected Q units."""
    n_transports = 0
    while (self.depot.repair_stock >= self.q_repair):
      self.depot.repair_stock -= self.q_repair
      self.warehouse.get_repairables(self.q_repair)
      n_transports += 1

    # Log transport
    if n_transports > 0:
      self.event_data = self.event_data.append({
        'time': self.time,
        'event': 'repair_shipment',
        'quantity': n_transports
      }, ignore_index=True)
  

  def sim_step(self):
    """Simulate one time step for all entities (in order). In the first sim
    we use the following order:
      1) Demand is handled at depot;
      2) Repairable units are processed in warehouse;
      3) Ordering policies are checked.
    """
    # Step 1 and 2)
    self.depot.demand_step()
    self.warehouse.repair_step()

    # Step 3)
    self.policy_upstream()
    self.policy_downstream()

    # Log data
    self.depot.log()
    self.warehouse.log()

    # Increase time step
    self.time += 1

  def add_cost_column(self, stock_info):
    """Cost at certain time period"""
    # Reshape the event data such that each row consistutes one time period
    event_data = self.event_data.pivot(index='time', columns='event', values='quantity')

    # Add order events to dataset and set NA values to 0
    stock_info = stock_info.merge(event_data, how="left", on="time")
    stock_info = stock_info.fillna(value={'repair_shipment':0, 'service_order':0})

    # Add cost of service order
    stock_info['order_cost_service'] = stock_info.service_orders * self.c_service

    # Add cost of repair shpiment
    stock_info['order_cost_repair'] = stock_info.repair_shipment * self.c_repair
    
    # Add holding cost serviceables
    stock_info['hold_cost_serviceables'] = (stock_info.service_stock_depot.clip(lower=0) + stock_info.service_stock_warehouse.clip(lower=0)) * self.h_service

    # Add back order cost serviceables
    stock_info['back_cost_serviceables'] = stock_info.service_back_orders * self.b_service

    # Add holding cost repairables
    stock_info['hold_cost_repairables'] = (stock_info.service_stock_depot + stock_info.service_stock_warehouse) * self.h_repair

    # Combine all costs
    stock_info['costs'] = stock_info.order_cost_service + stock_info.order_cost_repair + stock_info.hold_cost_serviceables + stock_info.back_cost_serviceables + stock_info.hold_cost_repairables

    return stock_info

  
  def create_output_df(self):
    """Collect inventory levels and positions of models."""
    depot_data = self.depot.log_data
    depot_data['time'] = depot_data.index
    warehouse_data = self.warehouse.log_data
    warehouse_data['time'] = warehouse_data.index

    stock_info = depot_data.merge(warehouse_data, how='inner', on='time',
                                  suffixes=('_depot', '_warehouse'))

    # Add cost per unit time
    stock_info = self.add_cost_column(stock_info)

    return stock_info


  def save(self):
    """Save info of simulation"""
    # Save inventory information
    self.depot.save()
    self.warehouse.save()

    # Save event data
    self.event_data.pivot(index='time', columns='event', values='quantity').to_csv('output/event_data.csv')



class Depot(object):
  """Depot class that uses units and sends repairs.

  Attributes:
    service_stock: Net stock of servicable units.
    repair_stock: Net stock of repairable units.
    out_server: Server to send units to.
  """

  def __init__(self, demand_rate, init_service_stock):
    """Initialise Depot class."""

    # Inventory levels
    self.service_stock = init_service_stock
    self.service_stock_order = 0
    self.service_back_orders = 0
    self.repair_stock_level = 0
    self.repair_stock = 0

    # Demand rate
    self.demand_rate = demand_rate

    self.log_data = pd.DataFrame(columns=['service_stock', 'service_orders',
    'service_back_orders', 'service_stock_position', 'repair_stock'])
  
  def save(self):
    """Output log file to CSV"""
    self.log_data.to_csv('output/depot_output.csv')

  def demand_process(self):
    """Demand for one timestep.
    
    TODO implement random process."""
    return self.demand_rate
    
  def demand_step(self):
    """Let demand arrive and processes both stocks."""
    
    # Let demand arrive (for now deterministic)
    demand_size = self.demand_process()

    # Repairable stock increases with demand and service stock lowers
    self.repair_stock += demand_size

    if demand_size > self.service_stock:
      self.service_stock = 0
      self.service_back_orders += demand_size - self.service_stock
    else:
      self.service_stock -= demand_size

  def place_order(self, order_size):
    """Make an order of size Q"""
    self.service_stock_order += order_size

  def get_serviceables(self, batch_size):
    """Process incoming servicables."""

    # If there are back-orders, serve these first
    if self.service_back_orders > 0:
      batch_back = min(self.service_back_orders, batch_size)
      self.service_back_orders -= batch_back

    self.service_stock += batch_size - batch_back
    self.service_stock_order -= batch_size

  def get_service_inventory_position(self):
    return self.service_stock + self.service_stock_order - self.service_back_orders

  def get_repair_inventory_position(self):
    return self.repair_stock

  def log(self):
    """Log relevant info of simulation."""
    self.log_data = self.log_data.append({'service_stock': self.service_stock,
                                          'service_orders': self.service_stock_order,
                                          'service_back_orders': self.service_back_orders,
                                          'service_stock_position': self.get_service_inventory_position(),
                                          'repair_stock': self.repair_stock}, ignore_index=True)


class Warehouse(object):
  """Warehouse class that repairs repairable units and sends servicable units to
  customer.
  
  Attributes:
    service_stock: Net stock of servicable units.
    repair_stock: Net stock of repairable units.
    out_server: Server to send units to.
  """

  def __init__(self, repair_rate, init_service_stock):
    """Initialise Warehouse class."""
    self.service_stock = init_service_stock
    self.repair_stock = 0
    self.repair_rate = repair_rate

    self.log_data = pd.DataFrame(columns=['service_stock', 'repair_stock'])

  def save(self):
    """Output logger df to file"""
    self.log_data.to_csv('output/warehouse_output.csv')

  def get_repairables(self, nb_items):
    """Add incoming repairables to stock."""
    self.repair_stock += nb_items

  def get_order(self, n_units, order_size):
    """Send service stock to depot if available."""
    n_orders = int(self.service_stock / order_size)
    order_quantity = n_orders * order_size

    if n_orders > 0:
      self.service_stock -= order_quantity

    return order_quantity

  def repair_step(self):
    """Handle repair station at time step."""

    # Repair size is minimum of repair stock and repair rate
    repair_size = min(self.repair_stock, self.repair_rate)

    # Update repairable stocks and servicable stocks
    self.repair_stock -= repair_size
    self.service_stock += repair_size

  def log(self):
    """Log relevant info of simulation."""
    self.log_data = self.log_data.append({'service_stock': self.service_stock, 
                                          'repair_stock': self.repair_stock}, 
                                          ignore_index=True)

