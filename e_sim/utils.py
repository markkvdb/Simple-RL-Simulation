import pandas as pd
from itertools import product

from .sim_components import Simulator


def experiment_runner(settings, sim_time):
  """
  Runs all model experiments with all different combinations of provided 
  settings values.

  Keyword arguments:
    settings -- Dictionary with all settings
  """
  
  # Create all possible combinations of setting values
  settings_names = sorted(settings)
  settings_comb = list(product(*(settings[name] for name in settings_names)))

  sim_dfs = pd.DataFrame()

  # Run experiment for every combination
  for settings_vals in settings_comb:
    settings_experiment = dict(zip(settings_names, settings_vals))

    # Run simulation
    simulator = Simulator(sim_time, settings_experiment)
    simulator.run()
            
    # Save output to master data frame
    sim_data = simulator.create_output_df()
    for setting, value in settings_experiment.items():
      sim_data[setting] = value

    # Add column containing all settings as string
    setting_str = [name + '=' + str(value) for name, value in settings_experiment.items()]
    sim_data['settings'] = ', '.join(setting_str)

    sim_dfs = sim_dfs.append(sim_data)
  
  return(sim_dfs)




