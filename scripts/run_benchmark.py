from tensor_power_flow.ficional_grid_generator import generate_fictional_grid
from tensor_power_flow import TensorPowerFlow

# n_node_per_feeder = 10
# n_feeder = 100
# n_step = 1000
n_node_per_feeder = 3
n_feeder = 2
n_step = 10

cable_length_km_min = 0.8
cable_length_km_max = 1.2
load_p_w_max = 0.4e6 * 1.2
load_p_w_min = 0.4e6 * 0.8
pf = 0.95

load_scaling_min = 0.5
load_scaling_max = 1.5

fictional_dataset = generate_fictional_grid(
    n_node_per_feeder=3,
    n_feeder=2,
    cable_length_km_min=cable_length_km_min,
    cable_length_km_max=cable_length_km_max,
    load_p_w_max=load_p_w_max,
    load_p_w_min=load_p_w_min,
    pf=pf,
    n_step=n_step,
    load_scaling_min=load_scaling_min,
    load_scaling_max=load_scaling_max,
)

tpf = TensorPowerFlow(input_data=fictional_dataset["pgm_dataset"], system_frequency=50.0)

tpf.calculate_power_flow(update_data=fictional_dataset["pgm_update_dataset"], max_iteration=100, error_tolerance=1e-6)
