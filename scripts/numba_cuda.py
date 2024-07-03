import numba.cuda as cuda
import numpy as np
import time


@cuda.jit
def add_conjugate_kernel(ad, bd, cd):
    i = cuda.grid(1)
    if i < ad.shape[0]:
        ad[i] += bd[i] + cd[i].conjugate()


def add_conjugate_cuda(b, c, n_iter: int):
    ad = cuda.device_array_like(b)
    threadsperblock = 32
    bd = cuda.to_device(b)
    cd = cuda.to_device(c)

    step, size = ad.shape
    blockspergrid = (step + (threadsperblock - 1)) // threadsperblock

    for _ in range(n_iter):
        for i in range(size):
            add_conjugate_kernel[blockspergrid, threadsperblock](ad[:, i], bd[:, i], cd[:, i])
    return ad.copy_to_host()


def add_conjugate_numpy(b, c, n_iter: int):
    a = np.zeros_like(b)
    for _ in range(n_iter):
        a += b + c.conjugate()
    return a


def rnd_complex(shape, seed=0):
    rng = np.random.default_rng(seed=seed)
    return np.asfortranarray(
        rng.uniform(high=1.0, low=0.0, size=shape) + 1j * rng.uniform(high=1.0, low=0.0, size=shape)
    )


def run_test(size, step, n_iter=100, print_output=False):
    shape = (step, size)
    b = rnd_complex(shape, 0)
    c = rnd_complex(shape, 1)

    start_cuda = time.time()
    a_cuda = add_conjugate_cuda(b, c, n_iter=n_iter)
    end_cuda = time.time()

    start_numpy = time.time()
    a_numpy = add_conjugate_numpy(b, c, n_iter=n_iter)
    end_numpy = time.time()

    if print_output:
        print(f"step: {step}, size: {size}")
        print(f"Time taken for CUDA: {end_cuda - start_cuda} seconds")
        print(f"Time taken for NumPy: {end_numpy - start_numpy} seconds")
        diff = np.max(np.abs(a_cuda - a_numpy))
        print(f"Max diff: {diff}")


if __name__ == "__main__":
    run_test(size=10, step=10_000, print_output=False)
    run_test(size=10, step=1_000_000, print_output=True)
