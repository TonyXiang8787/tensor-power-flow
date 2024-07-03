import numba
import numba.cuda as cuda
import numpy as np


@cuda.jit
def add_conjugate(ad, bd, cd):
    i = cuda.grid(1)
    if i < ad.shape[0]:
        ad[i] = bd[i] + cd[i].conjugate()


def add_conjugate_batch(ad, bd, cd):
    threadsperblock = 32
    step, size = ad.shape
    blockspergrid = (step + (threadsperblock - 1)) // threadsperblock
    for i in range(size):
        print(ad[:, i])
        add_conjugate[blockspergrid, threadsperblock](ad[:, i], bd[:, i], cd[:, i])


def run_test():
    size = 100
    step = 10000
    shape = (step, size)
    ad = cuda.device_array(shape=shape, dtype=np.complex128, order="F")
    b = np.random.random(shape) + 1j * np.random.random(shape)
    c = np.random.random(shape) + 1j * np.random.random(shape)
    bd = cuda.to_device(b)
    cd = cuda.to_device(c)

    add_conjugate_batch(ad, bd, cd)

    a = ad.copy_to_host()
    print(np.max(np.abs(a - (b + c.conjugate()))))


if __name__ == "__main__":
    run_test()
