# Author: Mark van der Broek
# Date: 05-08-2019
# Description: Simple reverse logistic simulation

from sys import exit
from enum import Enum, auto
from heapq import heappush, heappop
from dataclasses import dataclass, field

import numpy as np
from numpy.random import Generator, PCG64
import pandas as pd
from tqdm import tqdm


# All events in our model
class Events(Enum):
    DEMAND = 1
    REPAIR = 2
    SHIP_REPAIR = 3
    SHIP_SERVICE = 4

    def __lt__(self, other):
        return self.value < other.value


@dataclass(order=True)
class EventItem:
    time: float
    event_type: Events
    sz: int

    def to_series(self):
        return pd.Series(
            {"time": self.time, "event": self.event_type.name, "quantity": self.sz}
        )


class EventQueue(object):
    """Wrapper for heapq. Handles pushing, popping and searching items."""

    def __init__(self):
        self.eq = []
        self.event_types = {}

        # Add event_types
        for event in Events:
            self.event_types[event] = 0

    def push(self, event: EventItem):
        heappush(self.eq, event)
        self.event_types[event.event_type] += 1

    def push2(self, time: float, event_type: Events, sz: int):
        heappush(self.eq, EventItem(time, event_type, sz))
        self.event_types[event_type] += 1

    def pop(self):
        event = heappop(self.eq)
        self.event_types[event.event_type] -= 1
        return event

    def find(self, event_type: Events):
        return self.event_types[event_type]


class Simulator(object):
    """The Simulator class initiates and updates all entities in the simulation

  Attributes:
    sim_entities: Array of all entities of the simulation (in order)
    delta_time: Fraction of one unit of time for step sz
    sim_time: Total units of time for simulation

  Methods:
    initialise: 
  """

    def __init__(self, sim_time: float, settings: dict, seed: int = 0):
        """Initialise Simulator class"""
        # Time of simulation attributes
        self.sim_time = sim_time
        self.seed = seed

        # Inventory model
        self.model = InventoryModel(
            Q_service=settings["Q_service"],
            Q_repair=settings["Q_repair"],
            S_depot=settings["S_depot"],
            S_warehouse=settings["S_warehouse"],
            demand_rate=settings["demand_rate"],
            demand_type=settings["demand_type"],
            repair_rate=settings["repair_rate"],
            repair_type=settings["repair_type"],
            init_stock_depot=settings["init_stock_depot"],
            init_stock_warehouse=settings["init_stock_warehouse"],
            seed=seed,
        )

    def run(self):
        """Run simulation"""
        self.model.init_queue()

        while self.model.time < self.sim_time:
            self.model.sim_step()

    def create_output_df(self):
        return self.model.create_output_df()

    def save(self):
        """After the simulation is complete, create a neat data frame"""
        self.model.save()


class InventoryModel(object):
    """This object holds all the entities of the inventory model and its interactions.
  
  Attributes:
    entities: list of entities used in the simulation
  """

    def __init__(
        self,
        Q_service: int,
        Q_repair: int,
        S_depot: int,
        S_warehouse: int,
        demand_rate: float,
        demand_type: str,
        repair_rate: float,
        repair_type: str,
        init_stock_depot: int,
        init_stock_warehouse: int,
        seed: int,
    ):
        """Initialise InventoryModel class."""
        # Random seed
        self.seed = seed

        # Entities of model
        self.depot = Depot(demand_rate, demand_type, init_stock_depot, seed)
        self.warehouse = Warehouse(repair_rate, repair_type, init_stock_warehouse, seed)

        # Policy settings
        self.q_service = Q_service
        self.q_repair = Q_repair
        self.s_depot = S_depot
        self.s_warehouse = S_warehouse

        # Transportation info
        self.shipped_service = 0
        self.shipped_repair = 0
        self.t_transport = 1

        # Logging info
        self.time = 0.0
        self.event_data = pd.DataFrame(columns=["time", "event", "quantity"])

        # Event queue
        self.eq = EventQueue()

    def init_queue(self):
        """Initialise the model with a demand (and potentially a repair)."""
        self._new_demand()
        self._new_repair()

    def handle_event(self, event: EventItem):
        """Handles repair and demand event"""
        event_type = event.event_type
        sz = event.sz

        if event_type == Events.DEMAND:
            self._handle_event_demand(sz)
        elif event_type == Events.REPAIR:
            self._handle_event_repair(sz)
        elif event_type == Events.SHIP_REPAIR:
            self._handle_event_shiprepair(sz)
        elif event_type == Events.SHIP_SERVICE:
            self._handle_event_shipservice(sz)

        # Log event
        self.event_data = self.event_data.append(event.to_series(), ignore_index=True)

    def _handle_event_demand(self, sz: int):
        """Update stocking levels and create new demand event"""
        self.depot.process_demand(sz)
        self._new_demand()

    def _handle_event_repair(self, sz: int):
        """Update stocking levels and (potentially create new repair event)"""
        self.warehouse.process_repair(sz)
        self._new_repair()

    def _handle_event_shiprepair(self, sz: int):
        """Handle new shipment to warehouse of repairable items."""
        self.shipped_repair -= sz
        self.warehouse.get_repairables(sz)
        self._new_repair()

    def _handle_event_shipservice(self, sz: int):
        """Handle new shipment to warehouse of serviceable items"""
        self.shipped_service -= sz
        self.depot.get_serviceables(sz)

    def _new_demand(self):
        """Create new demand (take from demand interarrival time distribution) and add to event queue."""
        dt, sz = self.depot.create_demand()
        self.eq.push2(self.time + dt, Events.DEMAND, sz)

    def _new_repair(self):
        """Unlike demand, repairs can only be issued when repair stock is 
    available at the warehouse and currently no item is being repaired."""
        if self.warehouse.repair_stock > 0 and self.warehouse.items_in_repair == 0:
            dt, sz = self.warehouse.create_repair()
            self.eq.push2(self.time + dt, Events.REPAIR, sz)

    def policy_upstream(self):
        """Order policy of the depot. Once inventory position drops below
    S, we initiate an order of sz Q."""
        # Place orders as long as inventory position is below S
        while self.depot.get_service_inventory_position() < self.s_depot:
            self.depot.place_order(self.q_service)

        # Send orders if they are available
        order_sz = self.warehouse.get_order(
            self.depot.service_stock_order, self.q_service
        )

        if order_sz > 0:
            self.shipped_service += order_sz
            self.eq.push2(self.time + self.t_transport, Events.SHIP_SERVICE, order_sz)

    def policy_downstream(self):
        """Send repairable units to warehouse once we collected Q units."""
        n = int(self.depot.repair_stock / self.q_repair)
        order_sz = n * self.q_repair

        if order_sz > 0:
            self.depot.repair_stock -= order_sz
            self.shipped_repair += order_sz
            self.eq.push2(self.time + self.t_transport, Events.SHIP_REPAIR, order_sz)

    def sim_step(self):
        """Simulate one time step for all entities (in order). In the first sim
    we use the following order:
      1) Demand is handled at depot;
      2) Repairable units are processed in warehouse;
      3) Ordering policies are checked.
    """
        # Get first event from queue
        event = self.eq.pop()

        # Handle event and update time
        self.time = event.time
        self.handle_event(event)

        # Check policies
        self.policy_upstream()
        self.policy_downstream()

        # Log data
        self.depot.log(self.time)
        self.warehouse.log(self.time)

    def create_output_df(self):
        """Collect inventory levels and positions of models."""
        # Sometimes multiple events happen at the same time period. We only want to
        # evaluate the cost at the end of the time period. Hence we group the
        # data by time and only keep the last observation of time.
        depot_data = self.depot.log_data.groupby("time").tail(1)
        warehouse_data = self.warehouse.log_data.groupby("time").tail(1)

        # Merge info of both locations
        stock_info = depot_data.merge(
            warehouse_data, how="inner", on="time", suffixes=("_depot", "_warehouse")
        )

        # And add the event data
        event_data = self.event_data.pivot_table(
            index="time",
            columns="event",
            values="quantity",
            aggfunc="sum",
            fill_value=0,
        )
        stock_info = stock_info.merge(event_data, how="inner", on="time")

        return stock_info

    def save(self):
        """Save info of simulation"""
        # Save inventory information
        self.depot.save()
        self.warehouse.save()

        # Save event data
        self.event_data.pivot(index="time", columns="event", values="quantity").to_csv(
            "output/event_data.csv"
        )


class Depot(object):
    """Depot class that uses units and sends repairs.

  Attributes:
    service_stock: Net stock of servicable units.
    repair_stock: Net stock of repairable units.
    out_server: Server to send units to.
  """

    def __init__(self, 
                 demand_rate: float, 
                 demand_type: str, 
                 init_service_stock: int, 
                 seed: int):
        """Initialise Depot class."""
        self.seed = seed
        self.generator = Generator(PCG64(seed))

        # Inventory levels
        self.service_stock = init_service_stock
        self.service_stock_order = 0
        self.service_back_orders = 0
        self.repair_stock_level = 0
        self.repair_stock = 0

        # Demand rate
        self.demand_rate = demand_rate
        if demand_type == "deterministic":
            self.create_demand = self._det_demand
        elif demand_type == "poisson":
            self.create_demand = self._poisson_demand

        self.log_data = pd.DataFrame(
            columns=[
                "time",
                "service_stock",
                "service_orders",
                "service_back_orders",
                "service_stock_position",
                "repair_stock",
            ]
        )

    def save(self):
        """Output log file to CSV"""
        self.log_data.to_csv("output/depot_output.csv")

    def _det_demand(self):
        """Demand for one timestep. Give time to next demand and size"""
        return (self.demand_rate, 1)

    def _poisson_demand(self):
        """Poisson demand process. Interarrival times are exponential."""
        return (self.generator.exponential(self.demand_rate), 1)

    def process_demand(self, sz: int):
        """Let demand arrive and processes both stocks."""

        # Repairable stock increases with demand and service stock lowers
        self.repair_stock += sz

        if sz > self.service_stock:
            self.service_stock = 0
            self.service_back_orders += sz - self.service_stock
        else:
            self.service_stock -= sz

    def place_order(self, sz: int):
        """Make an order of sz Q"""
        self.service_stock_order += sz

    def get_serviceables(self, sz: int):
        """Process incoming serviceables. Note that stocking levels of the 
    warehouse have already been processed."""

        # If there are back-orders, serve these first
        back_sz = min(self.service_back_orders, sz)
        self.service_back_orders -= back_sz

        # Remaining stock goes to inventory
        self.service_stock += sz - back_sz
        self.service_stock_order -= sz

    def get_service_inventory_position(self):
        return self.service_stock + self.service_stock_order - self.service_back_orders

    def get_repair_inventory_position(self):
        return self.repair_stock

    def log(self, time: float):
        """Log relevant info of simulation."""
        self.log_data = self.log_data.append(
            {
                "time": time,
                "service_stock": self.service_stock,
                "service_orders": self.service_stock_order,
                "service_back_orders": self.service_back_orders,
                "service_stock_position": self.get_service_inventory_position(),
                "repair_stock": self.repair_stock,
            },
            ignore_index=True,
        )


class Warehouse(object):
    """Warehouse class that repairs repairable units and sends servicable units to customer.
  
  Attributes:
    service_stock: Net stock of servicable units.
    repair_stock: Net stock of repairable units.
    out_server: Server to send units to.
    items_in_repair: number of items currently in repair shop.
  """

    def __init__(self, 
                 repair_rate: float, 
                 repair_type: str, 
                 init_service_stock: int, 
                 seed: int):
        """Initialise Warehouse class."""
        self.seed = seed
        self.generator = Generator(PCG64(seed+1))

        self.service_stock = init_service_stock
        self.repair_stock = 0
        self.repair_rate = repair_rate
        self.items_in_repair = 0

        if repair_type == "deterministic":
            self._create_repair_sample = self._det_repair
        elif repair_type == "poisson":
            self._create_repair_sample = self._exp_repair

        self.log_data = pd.DataFrame(
            columns=["time", "service_stock", "repair_stock", "items_in_repair"]
        )

    def save(self):
        """Output logger df to file."""
        self.log_data.to_csv("output/warehouse_output.csv")

    def get_repairables(self, sz: int):
        """Add incoming repairables to stock."""
        self.repair_stock += sz

    def get_order(self, n_units: int, order_sz: int):
        """Send service stock to depot if available."""
        # Check how many orders are requested and available and send as much as we
        # can
        n_orders_requested = int(n_units / order_sz)
        n_orders_avail = int(self.service_stock / order_sz)
        n_orders = min(n_orders_requested, n_orders_avail)

        order_quantity = n_orders * order_sz

        if order_quantity > 0:
            self.service_stock -= order_quantity

        return order_quantity

    def process_repair(self, sz: int):
        """Handle repair station at time step."""
        self.items_in_repair -= sz
        self.service_stock += sz

    def create_repair(self):
        """Create new repair job"""
        self.repair_stock -= 1
        self.items_in_repair += 1
        return self._create_repair_sample()

    def _det_repair(self):
        return (self.repair_rate, 1)
    
    def _exp_repair(self):
        return (self.generator.exponential(self.repair_rate), 1)

    def log(self, time: float):
        """Log relevant info of simulation."""
        self.log_data = self.log_data.append(
            {
                "time": time,
                "service_stock": self.service_stock,
                "repair_stock": self.repair_stock,
                "items_in_repair": self.items_in_repair,
            },
            ignore_index=True,
        )

