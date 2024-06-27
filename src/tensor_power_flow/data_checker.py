import numpy as np
from power_grid_model.data_types import BatchArray, BatchDataset, DenseBatchArray, SingleArray, SingleDataset

SUPPORTED_COMPONENTS = {"line", "source", "sym_load", "node"}


def check_input(input_data: SingleDataset):
    all_components = set(input_data.keys())
    if all_components != SUPPORTED_COMPONENTS:
        raise ValueError(f"Unsupported components: {all_components - SUPPORTED_COMPONENTS}")
    _check_load(input_data["sym_load"])
    _check_source(input_data["source"])
    _check_line(input_data["line"])


def _check_load(load_array: SingleArray):
    if not np.all(load_array["status"] == 1):
        raise ValueError("All loads must be active")


def _check_source(source_array: SingleArray):
    if source_array.shape != (1,):
        raise ValueError("There must be exactly one source")
    if not np.all(source_array["status"] == 1):
        raise ValueError("All sources must be connected")


def _check_line(line_array: SingleArray):
    if not (np.all(line_array["from_status"] == 1) and np.all(line_array["to_status"] == 1)):
        raise ValueError("All lines must be connected")


def check_update(input_data: SingleDataset, update_data: BatchDataset):
    all_components = set(update_data.keys())
    if all_components != {"sym_load"}:
        raise ValueError(f"Unsupported components: {all_components - {"sym_load"}} in the update data!")
    _check_load_update(input_data["sym_load"], update_data["sym_load"])


def _check_load_update(load_input_array: SingleArray, load_array: BatchArray):
    if not isinstance(load_array, np.ndarray):
        raise ValueError("Load update must be a dense batch array!")
    if not np.all(load_array["status"] == -128):
        raise ValueError("All load status should not be changed in the update data!")
    if np.any(np.isnan(load_array["p_specified"])):
        raise ValueError("All load p_specified should be specified in the update data!")
    if np.any(np.isnan(load_array["q_specified"])):
        raise ValueError("All load q_specified should be specified in the update data!")
    if not np.all(load_input_array["id"].reshape(1, -1) == load_array["id"]):
        raise ValueError("The order of loads should not be changed in the update data!")
