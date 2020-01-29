# Simple-RL-Simulation

## Note for self (27-11-2019)

Started implement the event-based simulation. Things left:

- [x] File log file using event `enum`.
- [x] Create initialisation for simulation so that event_queue is filled.
- [x] Repair process is now implemented with single server. Repair rate does not depend on the number of available items.
- [x] Items in transport now have a separete variable member in the `Model` class.
- [x] Order level of serviceable items at the depot is not lowered when shipment is initiated but when order arrives. Does this make sense? **YES**
- [x] Plots are now constructed by connecting points, but should have L shape instead of line.
- [x] Cost computation needs reconsideration for event-based simulator.

## Note for self (28-11-2019)

Plots can be solved using `plt.step`

- [x] Find model settings which will give back-orders

## Note for self (02-12-2019)

- [x] Bug in experiment_runner which does not store results anymore when there is an event that never happes in the simulation.
- [x] Progress bar does not support multithreading yet.

## Note for self (04-12-2019)

- [x] Add sensitivity analysis to the cost parameters.

## Note for self (29-01-2020)

- [ ] Moinzadeh model will not work since orders are processed in batches. This is possible since the forward and reverse flows are identical.
- [ ] Distribution of the arrival process of repairable items is the superposition of the Erlang distribution.
