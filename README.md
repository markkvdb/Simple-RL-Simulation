# Simple-RL-Simulation

## Note for self (27-11-2019)

Started implement the event-based simulation. Things left:

- [x] File log file using event `enum`.
- [x] Create initialisation for simulation so that event_queue is filled.
- [ ] Repair process is now implemented with single server. Repair rate does not depend on the number of available items.
- [ ] Items in transport should be handled carefully.
- [ ] Order level of serviceable items at the depot is not lowered when shipment is initiated but when order arrives. Does this make sense?
- [ ] 