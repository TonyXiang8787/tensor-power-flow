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
