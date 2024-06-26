from power_grid_model import PowerGridModel
from power_grid_model.data_types import SingleDataset, BatchDataset


class TensorPowerFlow:
    _model: PowerGridModel

    def __init__(self, input_data: SingleDataset, system_frequency: float):
        self._model = PowerGridModel(input_data, system_frequency)
