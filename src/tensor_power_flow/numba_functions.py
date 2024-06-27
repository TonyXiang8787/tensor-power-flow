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
def set_rhs(rhs, load_pu, load_type, load_node, u, u_abs):
    rhs[...] = 0.0
    u_abs[...] = np.abs(u)
    for i in range(len(load_type)):
        node_i = load_node[i]
        type_i = load_type[i]
        if type_i == CONST_POWER:
            rhs[:, node_i] -= np.conj(load_pu[:, node_i] / u[:, node_i])
        elif type_i == CONST_CURRENT:
            rhs[:, node_i] -= np.conj(load_pu[:, node_i] * u_abs[:, node_i] / u[:, node_i])
        elif type_i == CONST_IMPEDANCE:
            # formula: conj(s * u_abs^2 / u) = conj(s * u * conj(u) / u) = conj(s * conj(u)) = conj(s) * u
            rhs[:, node_i] -= np.conj(load_pu[:, node_i]) * u[:, node_i]
