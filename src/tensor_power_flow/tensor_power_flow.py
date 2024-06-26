from power_grid_model import PowerGridModel
from power_grid_model.data_types import SingleDataset, BatchDataset
import numpy as np
from .data_checker import check_input
import scipy.sparse as sp
import scipy.sparse.csgraph as csg
from typing import Optional
from . import BASE_POWER


class TensorPowerFlow:
    _input_data: SingleDataset
    _model: PowerGridModel
    _u_rated: float
    _n_node: int
    _n_line: int
    _n_load: int
    _node_reordered: bool
    _node_reordered_to_org: Optional[np.ndarray]
    _node_org_to_reordered: Optional[np.ndarray]
    _line_node_from: np.ndarray
    _line_node_to: np.ndarray
    _load_node: np.ndarray
    _source_node: int

    def __init__(self, input_data: SingleDataset, system_frequency: float):
        self._input_data = input_data
        self._model = PowerGridModel(input_data, system_frequency)
        check_input(input_data)
        self._u_rated = input_data["node"]["u_rated"][0]
        self._n_node = len(input_data["node"])
        self._n_line = len(input_data["line"])
        self._n_load = len(input_data["sym_load"])
        self._node_reordered = False
        self._node_reordered_to_org = None
        self._node_org_to_reordered = None
        self._line_node_from = self._model.get_indexer("node", input_data["line"]["from_node"])
        self._line_node_to = self._model.get_indexer("node", input_data["line"]["to_node"])
        self._load_node = self._model.get_indexer("node", input_data["sym_load"]["node"])
        self._source_node = self._model.get_indexer("node", input_data["source"]["node"])[0]

    def _graph_reorder(self):
        edge_i = np.concatenate((self._line_node_from, self._line_node_to), axis=0)
        edge_j = np.concatenate((self._line_node_to, self._line_node_from), axis=0)
        connection_array = sp.csr_array(
            (np.ones(self._n_line * 2), (edge_i, edge_j)), shape=(self._n_node, self._n_node)
        )
        reordered_node = csg.depth_first_order(
            connection_array, i_start=self._source_node, directed=False, return_predecessors=False
        )
        if len(reordered_node) != self._n_node:
            raise ValueError("The graph is not connected!")
        reordered_node = reordered_node[::-1]
        self._node_reordered_to_org = reordered_node
        self._node_org_to_reordered = np.full(shape=(self._n_node,), dtype=np.int64, fill_value=-1)
        self._node_org_to_reordered[reordered_node] = np.arange(self._n_node, dtype=np.int64)
        self._line_node_from = self._node_org_to_reordered[self._line_node_from]
        self._line_node_to = self._node_org_to_reordered[self._line_node_to]
        self._load_node = self._node_org_to_reordered[self._load_node]
        self._source_node = self._node_org_to_reordered[self._source_node]
        assert self._source_node == self._n_node - 1
        self._node_reordered = True
