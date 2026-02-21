/*
    Modified from
    https://people.sc.fsu.edu/~jburkardt/cpp_src/jacobi_eigenvalue/jacobi_eigenvalue.html
*/

#pragma once

#include "builtin.h"

namespace wp {

inline CUDA_CALLABLE void diag_get_vector_9(int n, float *a, float *v) {
    int i;

    for (i = 0; i < n; i++)
    {
        v[i] = a[i + i * n];
    }

    return;
}

inline CUDA_CALLABLE void identity_9(int n, float *a){
    int i;
    int j;
    int k;

    k = 0;
    for (j = 0; j < n; j++)
    {
        for (i = 0; i < n; i++)
        {
            if (i == j)
            {
                a[k] = 1.0;
            }
            else
            {
                a[k] = 0.0;
            }
            k = k + 1;
        }
    }

    return;
}

inline CUDA_CALLABLE void jacobi_eigenvalue_9(int n, float *a, int it_max, float *v,
        float *d, int &it_num, int &rot_num){
//****************************************************************************80
//
//  Purpose:
//
//    JACOBI_EIGENVALUE carries out the Jacobi eigenvalue iteration.
//
//  Discussion:
//
//    This function computes the eigenvalues and eigenvectors of a
//    real symmetric matrix, using Rutishauser's modfications of the classical
//    Jacobi rotation method with threshold pivoting.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    17 September 2013
//
//  Author:
//
//    John Burkardt
//
//  Reference:
//
//    Gene Golub, Charles VanLoan,
//    Matrix Computations,
//    Third Edition,
//    Johns Hopkins, 1996,
//    ISBN: 0-8018-4513-X,
//    LC: QA188.G65.
//
//  Input:
//
//    int N, the order of the matrix.
//
//    float A[N*N], the matrix, which must be square, real,
//    and symmetric.
//
//    int IT_MAX, the maximum number of iterations.
//
//  Output:
//
//    float V[N*N], the matrix of eigenvectors.
//
//    float D[N], the eigenvalues, in descending order.
//
//    int &IT_NUM, the total number of iterations.
//
//    int &ROT_NUM, the total number of rotations.
//
    float bw[9];
    float c;
    float g;
    float gapq;
    float h;
    int i;
    int j;
    int k;
    int l;
    int m;
    int p;
    int q;
    float s;
    float t;
    float tau;
    float term;
    float termp;
    float termq;
    float theta;
    float thresh;
    float w;
    float zw[9];

    identity_9(n, v);

    diag_get_vector_9(n, a, d);

    for (i = 0; i < n; i++)
    {
        bw[i] = d[i];
        zw[i] = 0.0;
    }
    it_num = 0;
    rot_num = 0;

    while (it_num < it_max)
    {
        it_num = it_num + 1;
        //
        //  The convergence threshold is based on the size of the elements in
        //  the strict upper triangle of the matrix.
        //
        thresh = 0.0;
        for (j = 0; j < n; j++)
        {
            for (i = 0; i < j; i++)
            {
                thresh = thresh + a[i + j * n] * a[i + j * n];
            }
        }

        thresh = sqrt(thresh) / (float)(4 * n);

        if (thresh == 0.0)
        {
            break;
        }

        for (p = 0; p < n; p++)
        {
            for (q = p + 1; q < n; q++)
            {
                gapq = 10.0 * fabs(a[p + q * n]);
                termp = gapq + fabs(d[p]);
                termq = gapq + fabs(d[q]);
                //
                //  Annihilate tiny offdiagonal elements.
                //
                if (4 < it_num &&
                    termp == fabs(d[p]) &&
                    termq == fabs(d[q]))
                {
                    a[p + q * n] = 0.0;
                }
                //
                //  Otherwise, apply a rotation.
                //
                else if (thresh <= fabs(a[p + q * n]))
                {
                    h = d[q] - d[p];
                    term = fabs(h) + gapq;

                    if (term == fabs(h))
                    {
                        t = a[p + q * n] / h;
                    }
                    else
                    {
                        theta = 0.5 * h / a[p + q * n];
                        t = 1.0 / (fabs(theta) + sqrt(1.0 + theta * theta));
                        if (theta < 0.0)
                        {
                            t = -t;
                        }
                    }
                    c = 1.0 / sqrt(1.0 + t * t);
                    s = t * c;
                    tau = s / (1.0 + c);
                    h = t * a[p + q * n];
                    //
                    //  Accumulate corrections to diagonal elements.
                    //
                    zw[p] = zw[p] - h;
                    zw[q] = zw[q] + h;
                    d[p] = d[p] - h;
                    d[q] = d[q] + h;

                    a[p + q * n] = 0.0;
                    //
                    //  Rotate, using information from the upper triangle of A only.
                    //
                    for (j = 0; j < p; j++)
                    {
                        g = a[j + p * n];
                        h = a[j + q * n];
                        a[j + p * n] = g - s * (h + g * tau);
                        a[j + q * n] = h + s * (g - h * tau);
                    }

                    for (j = p + 1; j < q; j++)
                    {
                        g = a[p + j * n];
                        h = a[j + q * n];
                        a[p + j * n] = g - s * (h + g * tau);
                        a[j + q * n] = h + s * (g - h * tau);
                    }

                    for (j = q + 1; j < n; j++)
                    {
                        g = a[p + j * n];
                        h = a[q + j * n];
                        a[p + j * n] = g - s * (h + g * tau);
                        a[q + j * n] = h + s * (g - h * tau);
                    }
                    //
                    //  Accumulate information in the eigenvector matrix.
                    //
                    for (j = 0; j < n; j++)
                    {
                        g = v[j + p * n];
                        h = v[j + q * n];
                        v[j + p * n] = g - s * (h + g * tau);
                        v[j + q * n] = h + s * (g - h * tau);
                    }
                    rot_num = rot_num + 1;
                }
            }
        }

        for (i = 0; i < n; i++)
        {
            bw[i] = bw[i] + zw[i];
            d[i] = bw[i];
            zw[i] = 0.0;
        }
    }
    //
    //  Restore upper triangle of input matrix.
    //
    for (j = 0; j < n; j++)
    {
        for (i = 0; i < j; i++)
        {
            a[i + j * n] = a[j + i * n];
        }
    }
    //
    //  Ascending sort the eigenvalues and eigenvectors.
    //
    for (k = 0; k < n - 1; k++)
    {
        m = k;
        for (l = k + 1; l < n; l++)
        {
            if (d[l] < d[m])
            {
                m = l;
            }
        }

        if (m != k)
        {
            t = d[m];
            d[m] = d[k];
            d[k] = t;
            for (i = 0; i < n; i++)
            {
                w = v[i + m * n];
                v[i + m * n] = v[i + k * n];
                v[i + k * n] = w;
            }
        }
    }
}

inline CUDA_CALLABLE void spd_project(int n, float *a, int it_max) {
/*
    Project an n*n real symmetric matrix (stored contiguously in 1d array a)
    to a positive semi-definite matrix.
    Matrix a is modified in place.
*/
    float v[9*9], d[9*9];
    int it_num, rot_num;
    int i, j, k;

    jacobi_eigenvalue_9(n, a, it_max, v, d, it_num, rot_num);

//    printf("d = ");
//    for (i = 0; i < n; i++)
//        printf("%f%c", d[i], " \n"[i == n - 1]);
//
//    printf("v = ");
//    for (i = 0; i < n; i++)
//        for (j = 0; j < n; j++)
//            printf("%f%c", v[i * n + j], " \n"[j == n - 1]);

    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++)
            a[i * n + j] = 0.;

    for (k = 0; k < n; k++)
        if (d[k] > 0)
            for (i = 0; i < n; i++)
                for (j = 0; j < n; j++)
                    a[i * n + j] += d[k] * v[k * n + i] * v[k * n + j];
}

inline CUDA_CALLABLE void adj_spd_project(int n, float *a, int it_max, int adj_n, float *adj_a, int adj_it_max) {}

inline CUDA_CALLABLE void blocks_to_array(int n, array_t<mat_t<3,3,float> >& b, float *a) {
    int i, j, k, l;
    int b_size = b.shape[1];  // b.shape[0] = b.shape[1] >= n
    int n3 = n * 3;
    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++)
            for (k = 0; k < 3; k++)
                for (l = 0; l < 3; l++)
                    a[(i * 3 + k) * n3 + j * 3 + l] = b.data[i * b_size + j].data[k][l];
}

inline CUDA_CALLABLE void array_to_blocks(int n, array_t<mat_t<3,3,float> >& b, float *a) {
    int i, j, k, l, i_a, j_a;
    int b_size = b.shape[1];  // b.shape[0] = b.shape[1] >= n
    int n3 = n * 3;
    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++)
            for (k = 0; k < 3; k++)
                for (l = 0; l < 3; l++){
                    i_a = i * 3 + k;
                    j_a = j * 3 + l;
                    b.data[i * b_size + j].data[k][l] = (a[i_a * n3 + j_a] + a[j_a * n3 + i_a]) * 0.5; // Keep symmetric
                }
}

inline CUDA_CALLABLE void spd_project_blocks(int n, array_t<mat_t<3,3,float> >& b, int it_max) {
/*
    Only handle the the n*n blocks b[0:n, 0:n] (n <= 3).
*/
    float a[9*9];

    blocks_to_array(n, b, a);

    spd_project(3 * n, a, it_max);

    array_to_blocks(n, b, a);
}

inline CUDA_CALLABLE void adj_spd_project_blocks(int n, array_t<mat_t<3,3,float> >& b, int it_max,
        int adj_n, array_t<mat_t<3,3,float> >& adj_b, int adj_it_max) {}

}