from numba import njit
import numpy as np
from .base_power import BASE_POWER
from power_grid_model import LoadGenType

CONST_POWER = int(LoadGenType.const_power)
CONST_CURRENT = int(LoadGenType.const_current)
CONST_IMPEDANCE = int(LoadGenType.const_impedance)


@njit
def set_load_pu(load_pu: np.ndarray, p_array: np.ndarray, q_array: np.ndarray):
    load_pu[...] = (p_array + 1j * q_array) / BASE_POWER


@njit
def set_rhs(rhs, load_pu, load_type, load_node, u, u_abs, i_ref):
    rhs[...] = 0.0
    u_abs[...] = np.abs(u)
    for i in range(len(load_type)):
        node_i = load_node[i]
        type_i = load_type[i]
        if type_i == CONST_POWER:
            rhs[:, node_i] -= np.conj(load_pu[:, i] / u[:, node_i])
        elif type_i == CONST_CURRENT:
            rhs[:, node_i] -= np.conj(load_pu[:, i] * u_abs[:, node_i] / u[:, node_i])
        elif type_i == CONST_IMPEDANCE:
            # formula: conj(s * u_abs^2 / u) = conj(s * u * conj(u) / u) = conj(s * conj(u)) = conj(s) * u
            rhs[:, node_i] -= np.conj(load_pu[:, i]) * u[:, node_i]
    rhs[:, -1] += i_ref


@njit
def solve_rhs_inplace(indptr_l, indices_l, data_l, indptr_u, indices_u, data_u, rhs):
    size = rhs.shape[1]
    # forward substitution
    for i in range(size):
        for index_j in range(indptr_l[i], indptr_l[i + 1] - 1):
            j = indices_l[index_j]
            rhs[:, i] -= data_l[index_j] * rhs[:, j]
    # backward substitution
    for i in range(size - 1, -1, -1):
        for index_j in range(indptr_u[i + 1] - 1, indptr_u[i], -1):
            j = indices_u[index_j]
            rhs[:, i] -= data_u[index_j] * rhs[:, j]
        index_diag = indptr_u[i]
        rhs[:, i] /= data_u[index_diag]


@njit
def iterate_and_compare(u, rhs):
    size = u.shape[1]
    max_diff = 0.0
    for i in range(size):
        diff = np.max(np.abs(rhs[:, i] - u[:, i]))
        if diff > max_diff:
            max_diff = diff
        u[:, i] = rhs[:, i]
    return max_diff
