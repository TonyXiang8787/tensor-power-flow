import numba.cuda as cuda
import numpy as np
import time


@cuda.jit
def add_conjugate_kernel(ad, bd, cd, n_iter: int):
    step, size = ad.shape
    i = cuda.grid(1)
    if i >= step:
        return

    for _ in range(n_iter):
        for j in range(size):
            ad[i, j] += bd[i, j] + cd[i, j].conjugate()


def add_conjugate_cuda(b, c, n_iter: int):
    ad = cuda.device_array_like(b)
    assert ad.is_c_contiguous() == b.flags["C_CONTIGUOUS"]
    assert ad.is_f_contiguous() == b.flags["F_CONTIGUOUS"]
    bd = cuda.to_device(b)
    cd = cuda.to_device(c)

    threadsperblock = 32
    step, _ = ad.shape
    blockspergrid = (step + (threadsperblock - 1)) // threadsperblock
    add_conjugate_kernel[blockspergrid, threadsperblock](ad, bd, cd, n_iter)
    return ad.copy_to_host()


def add_conjugate_numpy(b, c, n_iter: int):
    a = np.zeros_like(b)
    for _ in range(n_iter):
        a += b + c.conjugate()
    return a


def rnd_complex(shape, seed=0, c_contiguous=True):
    rng = np.random.default_rng(seed=seed)
    a = rng.uniform(high=1.0, low=0.0, size=shape) + 1j * rng.uniform(high=1.0, low=0.0, size=shape)
    if not c_contiguous:
        return np.asfortranarray(a)
    return a


def run_test(size, step, n_iter=100, print_output=False, c_contiguous=True):
    shape = (step, size)
    b = rnd_complex(shape, seed=0, c_contiguous=c_contiguous)
    c = rnd_complex(shape, seed=1, c_contiguous=c_contiguous)

    start_cuda = time.time()
    a_cuda = add_conjugate_cuda(b, c, n_iter=n_iter)
    end_cuda = time.time()

    start_numpy = time.time()
    a_numpy = add_conjugate_numpy(b, c, n_iter=n_iter)
    end_numpy = time.time()

    if print_output:
        print(f"step: {step}, size: {size}, c_contiguous: {c_contiguous}")
        print(f"Time taken for CUDA: {end_cuda - start_cuda} seconds")
        print(f"Time taken for NumPy: {end_numpy - start_numpy} seconds")
        diff = np.max(np.abs(a_cuda - a_numpy))
        print(f"Max diff: {diff}")


if __name__ == "__main__":
    # C order
    # run_test(size=10, step=10_000, print_output=False, c_contiguous=True)
    # run_test(size=100, step=100_000, print_output=True, c_contiguous=True)
    # run_test(size=10, step=1_000_000, print_output=True, c_contiguous=True)
    
    # F order
    run_test(size=10, step=10_000, print_output=False, c_contiguous=False)
    run_test(size=100, step=100_000, print_output=True, c_contiguous=False)
    run_test(size=10, step=1_000_000, print_output=True, c_contiguous=False)
