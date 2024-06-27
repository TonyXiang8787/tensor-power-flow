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
    _system_frequency: float
    _model: PowerGridModel
    _u_rated: float
    _y_base: float
    _n_node: int
    _n_line: int
    _n_load: int
    _node_reordered_to_org: Optional[np.ndarray]
    _node_org_to_reordered: Optional[np.ndarray]
    _line_node_from: np.ndarray
    _line_node_to: np.ndarray
    _load_node: np.ndarray
    _source_node: int
    _y_bus: Optional[sp.csc_array]

    def __init__(self, input_data: SingleDataset, system_frequency: float):
        self._input_data = input_data
        self._system_frequency = system_frequency
        self._model = PowerGridModel(input_data, system_frequency)
        check_input(input_data)
        self._u_rated = input_data["node"]["u_rated"][0]
        self._y_base = BASE_POWER / (self._u_rated ** 2)
        self._n_node = len(input_data["node"])
        self._n_line = len(input_data["line"])
        self._n_load = len(input_data["sym_load"])
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

    def _build_y_bus(self):
        # branch admitance matrix
        y_series = 1.0 / (self._input_data["line"]["r1"] + 1j * self._input_data["line"]["x1"])
        y_shunt = (
            (0.5 * 2 * np.pi * self._system_frequency)
            * self._input_data["line"]["c1"]
            * (1j + self._input_data["line"]["tan1"])
        )
        all_y = np.concatenate((y_series, y_shunt, y_shunt), axis=0) / self._y_base
        y_branch = sp.dia_array((all_y, 0), shape=(self._n_line * 3, self._n_line * 3))

        # incidence matrix
        # series_from, series_to, shunt_from, shunt_to
        ones = np.ones(shape=(self._n_line,), dtype=np.int64)
        incidence_entry = np.concatenate((ones, -ones, ones, ones), axis=0)
        incidance_i = np.concatenate(
            (self._line_node_from, self._line_node_to, self._line_node_from, self._line_node_to), axis=0
        )
        incidance_j = np.concatenate(
            (
                np.arange(self._n_line),
                np.arange(self._n_line),
                np.arange(self._n_line, self._n_line * 2),
                np.arange(self._n_line * 2, self._n_line * 3),
            ),
            axis=0,
        )
        incidence_matrix = sp.csc_array(
            (incidence_entry, (incidance_i, incidance_j)), shape=(self._n_node, self._n_line * 3)
        )
        self._y_bus = (incidence_matrix @ y_branch @ incidence_matrix.T).tocsc(copy=True)

    def calculate_power_flow(
        self, *, update_data: BatchDataset, max_iteration: int = 20, error_tolerance: float = 1e-8
    ):
        if self._node_reordered_to_org is None:
            self._graph_reorder()
        if self._y_bus is None:
            self._build_y_bus()
        # diag_pivot_thresh = 0.0