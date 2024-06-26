from power_grid_model.data_types import SingleDataset, SingleArray
import numpy as np

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
