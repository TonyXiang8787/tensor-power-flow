from power_grid_model import PowerGridModel
from power_grid_model.data_types import SingleDataset, BatchDataset
import numpy as np
from .data_checker import check_input


class TensorPowerFlow:
    _input_data: SingleDataset
    _model: PowerGridModel
    _n_node: int
    _node_reorder: np.ndarray
    _line_node_from: np.ndarray
    _line_node_to: np.ndarray
    _load_node: np.ndarray
    _source_node: int

    def __init__(self, input_data: SingleDataset, system_frequency: float):
        self._input_data = input_data
        self._model = PowerGridModel(input_data, system_frequency)
        check_input(input_data)
        self._n_node = len(input_data["node"])
        self._node_reorder = np.zeros(self._n_node, dtype=np.int64)
        self._line_node_from = self._model.get_indexer("line", input_data["line"]["from_node"])
        self._line_node_to = self._model.get_indexer("line", input_data["line"]["to_node"])
        self._load_node = self._model.get_indexer("sym_load", input_data["sym_load"]["node"])
        self._source_node = self._model.get_indexer("source", input_data["source"]["node"])[0]
        self._graph_reorder()

    def _graph_reorder(self):
        pass
