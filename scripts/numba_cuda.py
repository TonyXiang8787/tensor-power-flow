import time

import numba
import numba.cuda as cuda
import numpy as np
from numba.core.errors import NumbaPerformanceWarning

import warnings

warnings.simplefilter("ignore", category=NumbaPerformanceWarning)


@cuda.jit
def add_conjugate_kernel(ad, bd, cd, indices_d, n_iter: int):
    step, _ = ad.shape
    i = cuda.grid(1)
    if i >= step:
        return

    for _ in range(n_iter):
        for j, index in enumerate(indices_d):
            ad[i, index] += bd[i, j] + cd[i, j].conjugate()


def add_conjugate_cuda(b, c, indices, n_iter: int):
    ad = cuda.device_array_like(b)
    assert ad.is_c_contiguous() == b.flags["C_CONTIGUOUS"]
    assert ad.is_f_contiguous() == b.flags["F_CONTIGUOUS"]
    bd = cuda.to_device(b)
    cd = cuda.to_device(c)
    indices_d = cuda.to_device(indices)

    threadsperblock = 32
    step, _ = ad.shape
    blockspergrid = (step + (threadsperblock - 1)) // threadsperblock
    add_conjugate_kernel[blockspergrid, threadsperblock](ad, bd, cd, indices_d, n_iter)
    return ad.copy_to_host()


@numba.njit(parallel=True)
def add_conjugate_numba_cpu(b, c, indices, n_iter: int):
    a = np.zeros_like(b)
    for _ in range(n_iter):
        for i, index in enumerate(indices):
            a[:, index] += b[:, i] + c[:, i].conjugate()
    return a


def rng_array(rng, shape):
    return rng.uniform(high=1.0, low=0.0, size=shape).astype(np.float64)


def rnd_complex(shape, seed=0):
    step, size = shape
    shape_bc = (step, size * 2)
    rng = np.random.default_rng(seed=seed)
    b = rng_array(rng, shape_bc) + 1j * rng_array(rng, shape_bc)
    c = rng_array(rng, shape_bc) + 1j * rng_array(rng, shape_bc)
    b = np.asfortranarray(b)
    c = np.asfortranarray(c)
    indices = rng.integers(low=0, high=size, size=size * 2, dtype=np.int64)
    return b, c, indices


def run_test(size, step, n_iter=100, print_output=False):
    shape = (step, size)
    b, c, indices = rnd_complex(shape)

    start_cuda = time.time()
    a_cuda = add_conjugate_cuda(b, c, n_iter=n_iter, indices=indices)
    end_cuda = time.time()

    start_numba_cpu = time.time()
    a_numba_cpu = add_conjugate_numba_cpu(b, c, n_iter=n_iter, indices=indices)
    end_numba_cpu = time.time()

    if print_output:
        print(f"step: {step}, size: {size}")
        print(f"Time taken for CUDA: {end_cuda - start_cuda} seconds")
        print(f"Time taken for Numba CPU: {end_numba_cpu - start_numba_cpu} seconds")
        diff = np.max(np.abs(a_cuda - a_numba_cpu))
        print(f"Max diff: {diff}")


if __name__ == "__main__":
    numba.set_num_threads(4)
    run_test(size=10, step=10_000, print_output=False)
    run_test(size=100, step=100_000, print_output=True)
    run_test(size=10, step=1_000_000, print_output=True)
