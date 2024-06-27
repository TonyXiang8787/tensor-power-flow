from tensor_power_flow.ficional_grid_generator import generate_fictional_grid
from tensor_power_flow import TensorPowerFlow
from power_grid_model import PowerGridModel, CalculationMethod
import numpy as np
import time


def run_benchmark(n_node_per_feeder, n_feeder, n_step, print_result: bool = False):
    cable_length_km_min = 0.8
    cable_length_km_max = 1.2
    load_p_w_max = 0.4e6 * 1.2
    load_p_w_min = 0.4e6 * 0.8
    pf = 0.95

    load_scaling_min = 0.5
    load_scaling_max = 1.5

    fictional_dataset = generate_fictional_grid(
        n_node_per_feeder=n_node_per_feeder,
        n_feeder=n_feeder,
        cable_length_km_min=cable_length_km_min,
        cable_length_km_max=cable_length_km_max,
        load_p_w_max=load_p_w_max,
        load_p_w_min=load_p_w_min,
        pf=pf,
        n_step=n_step,
        load_scaling_min=load_scaling_min,
        load_scaling_max=load_scaling_max,
    )

    start_time = time.time()
    tpf = TensorPowerFlow(input_data=fictional_dataset["pgm_dataset"], system_frequency=50.0)
    tpf_result = tpf.calculate_power_flow(update_data=fictional_dataset["pgm_update_dataset"])
    end_time = time.time()
    tpf_time = end_time - start_time

    start_time = time.time()
    pgm = PowerGridModel(input_data=fictional_dataset["pgm_dataset"], system_frequency=50.0)
    pgm_result = pgm.calculate_power_flow(
        update_data=fictional_dataset["pgm_update_dataset"],
        output_component_types={"node"},
        calculation_method=CalculationMethod.iterative_current,
    )
    end_time = time.time()
    pgm_time = end_time - start_time

    max_diff = get_max_diff(tpf_result, pgm_result)

    if print_result:
        print(f"Max diff: {max_diff}")
        print(f"TPF time: {tpf_time}")
        print(f"PGM time: {pgm_time}")


def get_max_diff(tpf_result, pgm_result):
    u_tpf = tpf_result["node"]["u_pu"] * np.exp(1j * tpf_result["node"]["u_angle"])
    u_pgm = pgm_result["node"]["u_pu"] * np.exp(1j * pgm_result["node"]["u_angle"])
    return np.max(np.abs(u_tpf - u_pgm))


if __name__ == "__main__":
    run_benchmark(n_node_per_feeder=3, n_feeder=2, n_step=10, print_result=False)
    run_benchmark(n_node_per_feeder=10, n_feeder=100, n_step=10000, print_result=True)
