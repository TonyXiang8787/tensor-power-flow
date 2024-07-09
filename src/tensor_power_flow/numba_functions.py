import numpy as np
from numba import njit
from power_grid_model import LoadGenType

from .base_power import BASE_POWER

CONST_POWER = int(LoadGenType.const_power)
CONST_CURRENT = int(LoadGenType.const_current)
CONST_IMPEDANCE = int(LoadGenType.const_impedance)


@njit
def set_load_pu(load_pu: np.ndarray, p_array: np.ndarray, q_array: np.ndarray):
    load_pu[...] = (p_array + 1j * q_array) / BASE_POWER

def get_load_pu(load_profile):
    p_array = np.asfortranarray(load_profile["p_specified"])
    q_array = np.asfortranarray(load_profile["q_specified"])
    load_pu = np.empty(shape=p_array.shape, dtype=np.complex128, order="F")
    set_load_pu(load_pu, p_array, q_array)
    return load_pu


def set_rhs_impl(rhs, load_pu, load_type, load_node, u, i_ref):
    rhs[...] = 0.0
    for i in range(len(load_type)):
        node_i = load_node[i]
        type_i = load_type[i]
        if type_i == CONST_POWER:
            rhs[:, node_i] -= np.conj(load_pu[:, i] / u[:, node_i])
        elif type_i == CONST_CURRENT:
            rhs[:, node_i] -= np.conj(load_pu[:, i] * np.abs(u[:, node_i]) / u[:, node_i])
        elif type_i == CONST_IMPEDANCE:
            # formula: conj(s * u_abs^2 / u) = conj(s * u * conj(u) / u) = conj(s * conj(u)) = conj(s) * u
            rhs[:, node_i] -= np.conj(load_pu[:, i]) * u[:, node_i]
    rhs[:, -1] += i_ref


def solve_rhs_inplace_impl(indptr_l, indices_l, data_l, indptr_u, indices_u, data_u, rhs):
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


def iterate_and_compare_impl(u, rhs):
    size = u.shape[1]
    max_diff2 = 0.0
    for i in range(size):
        diff2_arr = rhs[:, i] - u[:, i]
        diff2 = np.max(diff2_arr.real**2 + diff2_arr.imag**2)
        if diff2 > max_diff2:
            max_diff2 = diff2
        u[:, i] = rhs[:, i]
    return max_diff2


set_rhs_seq = njit(set_rhs_impl)
solve_rhs_inplace_seq = njit(solve_rhs_inplace_impl)
iterate_and_compare_seq = njit(iterate_and_compare_impl)

set_rhs_parallel = njit(set_rhs_impl, parallel=True)
solve_rhs_inplace_parallel = njit(solve_rhs_inplace_impl, parallel=True)
iterate_and_compare_parallel = njit(iterate_and_compare_impl, parallel=True)
