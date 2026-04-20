from ml4cps import examples
from ml4cps import vis


discrete_data, continuous_data = examples.conveyor_system_sfowl()

vis.plot_timeseries(discrete_data).show('browser')
