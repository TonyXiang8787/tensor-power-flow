import time

import numpy as np
import numba.cuda as cuda
from power_grid_model import CalculationMethod, PowerGridModel

from tensor_power_flow import TensorPowerFlow
from tensor_power_flow.ficional_grid_generator import generate_fictional_grid

import cProfile
import pstats

from pathlib import Path

PROFILE_PATH = Path(__file__).parent / "profile_results"
PROFILE_PATH.mkdir(exist_ok=True, parents=True)


def run_benchmark(n_node_per_feeder, n_feeder, n_step, print_result: bool = False, threading: int = -1):
    enable_gpu = threading != -1 and cuda.is_available()

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

    pgm = PowerGridModel(input_data=fictional_dataset["pgm_dataset"], system_frequency=50.0)
    tpf = TensorPowerFlow(input_data=fictional_dataset["pgm_dataset"], system_frequency=50.0)

    profiler = cProfile.Profile()
    profiler.enable()
    start_time = time.time()
    tpf_result = tpf.calculate_power_flow(update_data=fictional_dataset["pgm_update_dataset"], threading=threading)
    end_time = time.time()
    tpf_time = end_time - start_time
    if print_result and enable_gpu:
        with open(PROFILE_PATH / f"node_{n_node_per_feeder * n_feeder}_step_{n_step}_tpf.stats", "w") as f:
            pstats.Stats(profiler, stream=f).sort_stats("cumulative").print_stats()

    start_time = time.time()
    pgm_result = pgm.calculate_power_flow(
        update_data=fictional_dataset["pgm_update_dataset"],
        output_component_types={"node"},
        calculation_method=CalculationMethod.iterative_current,
        threading=threading,
    )
    end_time = time.time()
    profiler.disable()
    pgm_time = end_time - start_time

    if enable_gpu:
        profiler = cProfile.Profile()
        profiler.enable()
        start_time = time.time()
        tpf_gpu_result = tpf.calculate_power_flow_gpu(update_data=fictional_dataset["pgm_update_dataset"])
        end_time = time.time()
        profiler.disable()
        tpf_gpu_time = end_time - start_time
        max_diff = get_max_diff(tpf_result, pgm_result, tpf_gpu_result)
        if print_result:
            with open(PROFILE_PATH / f"node_{n_node_per_feeder * n_feeder}_step_{n_step}_tpf_gpu.stats", "w") as f:
                pstats.Stats(profiler, stream=f).sort_stats("cumulative").print_stats()
    else:
        max_diff = get_max_diff(tpf_result, pgm_result)

    if print_result:
        print("Benchmark result:")
        print(f"Total number of nodes: {n_node_per_feeder * n_feeder}")
        print(f"Number of steps: {n_step}")
        print(f"Threading: {threading}")
        print(f"Max diff: {max_diff}")
        print(f"PGM time: {pgm_time}")
        print(f"TPF time: {tpf_time}")
        if enable_gpu:
            print(f"TPF GPU time: {tpf_gpu_time}")
        print("\n\n")


def get_max_diff(*results):
    u_results = [result["node"]["u_pu"] * np.exp(1j * result["node"]["u_angle"]) for result in results]
    max_diffs = [np.max(np.abs(r1 - r2)) for r1, r2 in zip(u_results[:-1], u_results[1:])]
    return np.max(max_diffs)


if __name__ == "__main__":
    # pre compile
    run_benchmark(n_node_per_feeder=3, n_feeder=2, n_step=10, print_result=False)
    run_benchmark(n_node_per_feeder=3, n_feeder=2, n_step=10, print_result=False, threading=4)

    run_benchmark(n_node_per_feeder=10, n_feeder=100, n_step=10_000, print_result=True)
    run_benchmark(n_node_per_feeder=10, n_feeder=10, n_step=100_000, print_result=True)
    run_benchmark(n_node_per_feeder=10, n_feeder=100, n_step=10_000, print_result=True, threading=4)
    run_benchmark(n_node_per_feeder=5, n_feeder=2, n_step=1_000_000, print_result=True, threading=4)
