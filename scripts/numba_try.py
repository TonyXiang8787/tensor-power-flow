from numba import njit
import numpy as np


@njit
def array_assignment(a, b, index):
    for i in range(len(index)):
        a[:, index[i]] += b[:, i]


def try_fortran():
    a = np.zeros(shape=(3, 5), order="F", dtype=np.float64)
    rng = np.random.default_rng(seed=0)
    b = np.asfortranarray(rng.random(size=(3, 10), dtype=np.float64))
    index = rng.integers(low=0, high=5, size=10)
    array_assignment(a, b, index)
    print(b)
    print(index)
    print(a)


if __name__ == "__main__":
    try_fortran()
