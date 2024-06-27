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
def set_rhs(rhs, load_pu, load_type, load_node, u_abs):
    for i, node_i, type_i in enumerate(zip(load_node, load_type)):
        if type_i == CONST_POWER:
            rhs[:, node_i] -= np.conj(load_pu[:, node_i])
        elif type_i == CONST_CURRENT:
            rhs[:, node_i] -= load_pu[:, node_i] * u_abs[:, node_i]
        elif type_i == CONST_IMPEDANCE:
            rhs[:, node_i] -= np.conj(load_pu[node_i]) / np.abs(load_pu[node_i] ** 2)