import numba.cuda as cuda
from .base_power import BASE_POWER
from power_grid_model import LoadGenType
import numpy as np

from numba.core.errors import NumbaPerformanceWarning

import warnings

warnings.simplefilter("ignore", category=NumbaPerformanceWarning)


CONST_POWER = int(LoadGenType.const_power)
CONST_CURRENT = int(LoadGenType.const_current)
CONST_IMPEDANCE = int(LoadGenType.const_impedance)
THREADS_PER_BLOCK = 32


@cuda.jit
def _set_load_pu(load_pu, p_array, q_array):
    step, size = p_array.shape
    i, j = cuda.grid(2)
    if i < step and j < size:
        load_pu[i, j] = (p_array[i, j] + 1j * q_array[i, j]) / BASE_POWER


def get_load_pu(load_profile):
    p_array = np.asfortranarray(load_profile["p_specified"])
    q_array = np.asfortranarray(load_profile["q_specified"])
    p_device = cuda.to_device(p_array)
    q_device = cuda.to_device(q_array)
    step, size = p_array.shape
    load_pu = cuda.device_array(shape=(step, size), dtype=np.complex128, order="F")

    threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK)
    blockspergrid_x = (step + (threadsperblock[0] - 1)) // threadsperblock[0]
    blockspergrid_y = (size + (threadsperblock[1] - 1)) // threadsperblock[1]
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    _set_load_pu[blockspergrid, threadsperblock](load_pu, p_device, q_device)
    return load_pu
