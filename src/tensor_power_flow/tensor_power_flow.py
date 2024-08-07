from typing import Optional

import numba
import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph as csg
import scipy.sparse.linalg as splin
from power_grid_model import PowerGridModel
from power_grid_model.data_types import BatchDataset, SingleDataset

from .base_power import BASE_POWER
from .data_checker import check_input, check_update
from . import numba_functions as nf
from . import numba_cuda_functions as ncf


class TensorPowerFlow:
    _input_data: SingleDataset
    _system_frequency: float
    _model: PowerGridModel
    _u_rated: float
    _y_base: float
    _n_node: int
    _n_line: int
    _n_load: int
    _node_reordered_to_org: Optional[np.ndarray] = None
    _node_org_to_reordered: Optional[np.ndarray] = None
    _line_node_from: np.ndarray
    _line_node_to: np.ndarray
    _load_node: np.ndarray
    _source_node: int
    _load_type: np.ndarray
    _y_bus: Optional[sp.csc_array] = None
    _l_matrix: Optional[sp.csr_array] = None
    _u_matrix: Optional[sp.csr_array] = None
    _y_ref: Optional[np.complex128] = None
    _u_ref: Optional[np.complex128] = None
    _i_ref: Optional[np.complex128] = None

    def __init__(self, input_data: SingleDataset, system_frequency: float):
        self._input_data = input_data
        self._system_frequency = system_frequency
        self._model = PowerGridModel(input_data, system_frequency)
        check_input(input_data)
        self._u_rated = input_data["node"]["u_rated"][0]
        self._y_base = BASE_POWER / (self._u_rated**2)
        self._n_node = len(input_data["node"])
        self._n_line = len(input_data["line"])
        self._n_load = len(input_data["sym_load"])
        self._line_node_from = self._model.get_indexer("node", input_data["line"]["from_node"])
        self._line_node_to = self._model.get_indexer("node", input_data["line"]["to_node"])
        self._load_node = self._model.get_indexer("node", input_data["sym_load"]["node"])
        self._source_node = self._model.get_indexer("node", input_data["source"]["node"])[0]
        self._load_type = input_data["sym_load"]["type"].copy()

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

    def _factorize_matrix(self):
        if self._y_bus is None:
            self._build_y_bus()
        matrix = self._y_bus.tocsc(copy=True)
        source = self._input_data["source"][0]
        z_source_abs = BASE_POWER / source["sk"]
        rx_ratio = source["rx_ratio"]
        x_source = z_source_abs / np.sqrt(1 + rx_ratio**2)
        r_source = x_source * rx_ratio
        z_source = r_source + 1j * x_source
        y_source = 1.0 / z_source
        matrix.data[-1] += y_source
        splu: splin.SuperLU = splin.splu(matrix, permc_spec="NATURAL", diag_pivot_thresh=0.0)
        if not np.all(splu.perm_r == np.arange(self._n_node)):
            raise ValueError("The row permutation is not correct!")
        if not np.all(splu.perm_c == np.arange(self._n_node)):
            raise ValueError("The column permutation is not correct!")
        self._l_matrix = splu.L.tocsr(copy=True)
        self._u_matrix = splu.U.tocsr(copy=True)
        self._y_ref = y_source

    def pre_cache_calculation(self):
        if self._node_reordered_to_org is None:
            self._graph_reorder()
        if self._y_bus is None:
            self._build_y_bus()
        if self._l_matrix is None:
            self._factorize_matrix()
        # u variable, flat start as u_ref
        self._u_ref = self._input_data["source"][0]["u_ref"] + 0.0 * 1j
        self._i_ref = self._y_ref * self._u_ref

    def calculate_power_flow(
        self, *, update_data: BatchDataset, max_iteration: int = 20, error_tolerance: float = 1e-8, threading=-1
    ):
        check_update(self._input_data, update_data)
        self.pre_cache_calculation()
        load_profile = update_data["sym_load"]
        n_steps = load_profile.shape[0]

        if threading == -1 or threading == 1:
            use_parallel = False
        elif threading > 1:
            numba.set_num_threads(threading)
            use_parallel = True
        else:
            numba.set_num_threads(numba.config.NUMBA_NUM_THREADS)
            use_parallel = True

        # initialize
        # load_pu
        load_pu = nf.get_load_pu(load_profile)
        u = np.full(shape=(n_steps, self._n_node), fill_value=self._u_ref, dtype=np.complex128, order="F")
        # rhs variable, empty
        rhs = np.empty(shape=(n_steps, self._n_node), dtype=np.complex128, order="F")

        # iterate
        for _ in range(max_iteration):
            if use_parallel:
                nf.set_rhs_parallel(rhs, load_pu, self._load_type, self._load_node, u, self._i_ref)
                nf.solve_rhs_inplace_parallel(
                    indptr_l=self._l_matrix.indptr,
                    indices_l=self._l_matrix.indices,
                    data_l=self._l_matrix.data,
                    indptr_u=self._u_matrix.indptr,
                    indices_u=self._u_matrix.indices,
                    data_u=self._u_matrix.data,
                    rhs=rhs,
                )
                max_diff2 = nf.iterate_and_compare_parallel(u, rhs)
            else:
                nf.set_rhs_seq(rhs, load_pu, self._load_type, self._load_node, u, self._i_ref)
                nf.solve_rhs_inplace_seq(
                    indptr_l=self._l_matrix.indptr,
                    indices_l=self._l_matrix.indices,
                    data_l=self._l_matrix.data,
                    indptr_u=self._u_matrix.indptr,
                    indices_u=self._u_matrix.indices,
                    data_u=self._u_matrix.data,
                    rhs=rhs,
                )
                max_diff2 = nf.iterate_and_compare_seq(u, rhs)
            if max_diff2 < error_tolerance**2:
                break
        else:
            raise ValueError(f"The power flow calculation does not converge! Max diff: {np.sqrt(max_diff2)}")

        # reorder back to original
        u = u[:, self._node_org_to_reordered]
        return {"node": {"u_pu": np.abs(u), "u_angle": np.angle(u)}}

    def calculate_power_flow_gpu(
        self, *, update_data: BatchDataset, max_iteration: int = 20, error_tolerance: float = 1e-8
    ):
        check_update(self._input_data, update_data)
        self.pre_cache_calculation()
        load_profile = update_data["sym_load"]
        load_pu = ncf.get_load_pu(load_profile)
        load_node, load_type = ncf.get_load_node_and_type(self._load_node, self._load_type)
        u, rhs, u_diff2 = ncf.get_u_rhs(load_profile.shape[0], self._n_node, self._u_ref)
        lu_factorization = ncf.get_lu_factorization(self._l_matrix, self._u_matrix)

        for _ in range(max_iteration):
            ncf.set_rhs(rhs, load_pu, load_type, load_node, u, self._i_ref)
            ncf.solve_rhs_inplace(lu_factorization, rhs)
            max_diff2 = ncf.iterate_and_compare(u, rhs, u_diff2)
            if max_diff2 < error_tolerance**2:
                break
        else:
            raise ValueError(f"The power flow calculation does not converge! Max diff: {np.sqrt(max_diff2)}")
        u_pu, u_angle = ncf.get_result(u, self._node_org_to_reordered)
        return {"node": {"u_pu": u_pu, "u_angle": u_angle}}
