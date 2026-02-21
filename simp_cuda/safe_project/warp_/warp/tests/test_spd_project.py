import numpy as np
import warp as wp
from warp.tests.test_base import *

wp.init()


@wp.kernel
def spd_project_kernel(
        n: int,
        mat9s: wp.array(dtype=float, ndim=2),
        it_max: int,
):
    tid = wp.tid()

    wp.spd_project(n, mat9s[tid], it_max)


def test_spd_project_01(device):
    n = 4

    A = np.array([
        [-4.0,  -30.0,    60.0,   -35.0,
         -30.0,  300.0,  -675.0,   420.0,
         60.0, -675.0,  -1620.0, -1050.0,
         -35.0,  420.0, -1050.0,   700.0],
    ])

    print(np.linalg.eig(A.reshape(n, n)))

    mat9s = wp.array(A.reshape(1, n*n), dtype=float, device=device)
    it_max = 10

    wp.launch(kernel=spd_project_kernel,
              dim=1,
              inputs=[n, mat9s, it_max],
              device=device)
    wp.synchronize()

    A_SPD = mat9s.numpy().reshape(n, n)

    print(A_SPD)

    print(np.linalg.eig(A_SPD))


@wp.kernel
def spd_project_blocks_kernel(
        n: int,
        b: wp.array(dtype=wp.mat33, ndim=3),
        it_max: int,
):
    tid = wp.tid()

    wp.spd_project_blocks(n, b[tid], it_max)


def test_spd_project_blocks_01(device):
    print("\n============== test_spd_project_blocks_01 ==============\n")

    n = 3
    b_size = 4
    it_max = 24

    A = np.zeros((3*b_size, 3*b_size))
    A[:3*n, :3*n] = np.random.randn(3*n, 3*n)
    for i in range(3*n):
        for j in range(i):
            A[i, j] = A[j, i]

    B = np.ascontiguousarray(A.copy().reshape(b_size, 3, b_size, 3).transpose(0, 2, 1, 3))
    # print("A[:3*n, :3*n] =")
    # print(A[:3*n, :3*n])
    # print("B =")
    # print(B)
    print("eig(A) =")
    print(np.linalg.eig(A[:3*n, :3*n]))

    B_wp = wp.array(B[None, ...], dtype=wp.mat33, device=device)
    # print("B_wp.shape =", B_wp.shape)

    wp.launch(kernel=spd_project_blocks_kernel,
              dim=1,
              inputs=[n, B_wp, it_max],
              device=device)
    wp.synchronize()

    A_new = np.ascontiguousarray(B_wp.numpy().squeeze(0).transpose(0, 2, 1, 3)).reshape(b_size*3, b_size*3)

    # print("A_new =")
    # print(A_new)

    print(np.linalg.eig(A_new[:3 * n, :3 * n]))


if __name__ == '__main__':
    np.random.seed(20010313)

    device = "cuda"
    # test_spd_project_01(device)
    test_spd_project_blocks_01(device)

