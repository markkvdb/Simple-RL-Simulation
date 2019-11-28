# Simple-RL-Simulation

## Note for self (27-11-2019)

Started implement the event-based simulation. Things left:

- [x] File log file using event `enum`.
- [x] Create initialisation for simulation so that event_queue is filled.
- [ ] Repair process is now implemented with single server. Repair rate does not depend on the number of available items.
- [x] Items in transport now have a separete variable member in the `Model` class.
- [ ] Order level of serviceable items at the depot is not lowered when shipment is initiated but when order arrives. Does this make sense?
- [x] Plots are now constructed by connecting points, but should have L shape instead of line.
- [x] Cost computation needs reconsideration for event-based simulator.

## Note for self (28-11-2019)

Plots can be solved using `plt.step`

- [ ] Find model settings which will give back-orders
