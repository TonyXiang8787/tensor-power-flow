import numba.cuda as cuda
from .base_power import BASE_POWER
from power_grid_model import LoadGenType
import numpy as np
from scipy.sparse import csr_array
import math

from numba.core.errors import NumbaPerformanceWarning

import warnings

warnings.simplefilter("ignore", category=NumbaPerformanceWarning)


CONST_POWER = int(LoadGenType.const_power)
CONST_CURRENT = int(LoadGenType.const_current)
CONST_IMPEDANCE = int(LoadGenType.const_impedance)
THREADS_PER_BLOCK = 32


def _get_2d_grid(step, size):
    threads_per_block = (THREADS_PER_BLOCK, 1)
    blocks_per_grid_x = (step + (threads_per_block[0] - 1)) // threads_per_block[0]
    blocks_per_grid_y = (size + (threads_per_block[1] - 1)) // threads_per_block[1]
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)
    return blocks_per_grid, threads_per_block


def _get_1d_grid(step):
    blocks_per_grid = (step + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK
    return blocks_per_grid, THREADS_PER_BLOCK


@cuda.jit
def _set_load_pu(load_pu, p_array, q_array):
    step, size = p_array.shape
    i, j = cuda.grid(2)
    if i < step and j < size:
        load_pu[i, j] = complex(p_array[i, j], q_array[i, j]) / BASE_POWER


def get_load_pu(load_profile):
    p_array = np.asfortranarray(load_profile["p_specified"])
    q_array = np.asfortranarray(load_profile["q_specified"])
    p_device = cuda.to_device(p_array)
    q_device = cuda.to_device(q_array)
    step, size = p_array.shape
    load_pu = cuda.device_array(shape=(step, size), dtype=np.complex128, order="F")

    _set_load_pu[*_get_2d_grid(step, size)](load_pu, p_device, q_device)
    return load_pu


@cuda.jit
def _set_u(u, u_ref):
    step, size = u.shape
    i, j = cuda.grid(2)
    if i < step and j < size:
        u[i, j] = u_ref


def get_u_rhs(step, size, u_ref):
    u = cuda.device_array(shape=(step, size), dtype=np.complex128, order="F")
    rhs = cuda.device_array(shape=(step, size), dtype=np.complex128, order="F")
    u_diff2 = cuda.device_array(shape=(step, size), dtype=np.float64, order="F")
    _set_u[*_get_2d_grid(step, size)](u, u_ref)
    return u, rhs, u_diff2


def get_load_node_and_type(load_node, load_type):
    return cuda.to_device(load_node), cuda.to_device(load_type)


def get_lu_factorization(l_matrix: csr_array, u_matrix: csr_array):
    return {
        "indptr_l": cuda.to_device(l_matrix.indptr),
        "indices_l": cuda.to_device(l_matrix.indices),
        "data_l": cuda.to_device(l_matrix.data),
        "indptr_u": cuda.to_device(u_matrix.indptr),
        "indices_u": cuda.to_device(u_matrix.indices),
        "data_u": cuda.to_device(u_matrix.data),
    }


@cuda.jit
def _set_rhs_zero(rhs):
    step, size = rhs.shape
    i, j = cuda.grid(2)
    if i < step and j < size:
        rhs[i, j] = 0.0


@cuda.jit
def _set_rhs(rhs, load_pu, load_type, load_node, u, i_ref):
    step, n_node = u.shape
    n_load = load_pu.shape[1]
    i = cuda.grid(1)
    if i >= n_load:
        return
    for j_load in range(step):
        j_node = load_node[i]
        single_type = load_type[i]
        if single_type == CONST_POWER:
            rhs[i, j_node] -= (load_pu[i, j_load] / u[i, j_node]).conjugate()
        elif single_type == CONST_CURRENT:
            rhs[i, j_node] -= (load_pu[i, j_load] * abs(u[i, j_node]) / u[i, j_node]).conjugate()
        elif single_type == CONST_IMPEDANCE:
            # formula: conj(s * u_abs^2 / u) = conj(s * u * conj(u) / u) = conj(s * conj(u)) = conj(s) * u
            rhs[i, j_node] -= load_pu[i, j_load].conjugate() * u[i, j_node]
    rhs[i, n_node - 1] += i_ref


def set_rhs(rhs, load_pu, load_type, load_node, u, i_ref):
    step, size_u = rhs.shape
    _set_rhs_zero[*_get_2d_grid(step, size_u)](rhs)
    _set_rhs[_get_1d_grid(step)](rhs, load_pu, load_type, load_node, u, i_ref)


@cuda.jit
def _solve_rhs_inplace(indptr_l, indices_l, data_l, indptr_u, indices_u, data_u, rhs):
    step, size = rhs.shape
    i = cuda.grid(1)
    if i >= step:
        return
    # forward substitution
    for row in range(size):
        for index_col in range(indptr_l[row], indptr_l[row + 1] - 1):
            col = indices_l[index_col]
            rhs[i, row] -= data_l[index_col] * rhs[i, col]
    # backward substitution
    for row in range(size - 1, -1, -1):
        for index_col in range(indptr_u[row + 1] - 1, indptr_u[row], -1):
            col = indices_u[index_col]
            rhs[i, row] -= data_u[index_col] * rhs[i, col]
        index_diag = indptr_u[row]
        rhs[i, row] /= data_u[index_diag]


def solve_rhs_inplace(lu_factorization, rhs):
    step, _ = rhs.shape
    _solve_rhs_inplace[_get_1d_grid(step)](
        lu_factorization["indptr_l"],
        lu_factorization["indices_l"],
        lu_factorization["data_l"],
        lu_factorization["indptr_u"],
        lu_factorization["indices_u"],
        lu_factorization["data_u"],
        rhs,
    )


@cuda.jit
def _iterate_and_diff(u, rhs, u_diff2):
    step, size = u.shape
    i, j = cuda.grid(2)
    if i < step and j < size:
        diff = rhs[i, j] - u[i, j]
        diff2 = diff.real**2 + diff.imag**2
        u[i, j] = rhs[i, j]
        u_diff2[i, j] = diff2


@cuda.reduce
def _max_diff2(a, b):
    return max(a, b)


def iterate_and_compare(u, rhs, u_diff2):
    step, size = u.shape
    _iterate_and_diff[*_get_2d_grid(step, size)](u, rhs, u_diff2)
    max_diff2 = _max_diff2(u_diff2.ravel(order="F"))
    return max_diff2


@cuda.jit
def _get_result(u, node_org_to_reordered, u_pu, u_angle):
    step, size = u.shape
    i, j = cuda.grid(2)
    if i < step and j < size:
        u_single = u[i, node_org_to_reordered[j]]
        u_pu[i, j] = abs(u_single)
        tan_theta = u_single.imag / u_single.real
        theta = math.atan(tan_theta)
        u_angle[i, j] = theta if u_single.real > 0.0 else -theta


def get_result(u, node_org_to_reordered):
    step, size = u.shape
    u_pu = cuda.device_array(shape=(step, size), dtype=np.float64, order="F")
    u_angle = cuda.device_array(shape=(step, size), dtype=np.float64, order="F")
    node_org_to_reordered_device = cuda.to_device(node_org_to_reordered)
    _get_result[*_get_2d_grid(step, size)](u, node_org_to_reordered_device, u_pu, u_angle)
    return u_pu.copy_to_host(), u_angle.copy_to_host()
