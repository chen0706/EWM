#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 32

__global__ void l1_t_kernel(float *__restrict__ c, const float * __restrict__ a, const float * __restrict__ b,
    const int M, const int N, const int D)
{
    int block_row = blockIdx.y, block_col = blockIdx.x;
    int row = threadIdx.y, col = threadIdx.x;

    float c_value = 0;

    #pragma unroll
    for (int m = 0; m < (D + BLOCK_SIZE - 1) / BLOCK_SIZE; m++)
    {
        __shared__ float a_block[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float b_block[BLOCK_SIZE][BLOCK_SIZE];

        if (m * BLOCK_SIZE + col < D && block_row * BLOCK_SIZE + row < M)
        {
            a_block[row][col] = a[(m * BLOCK_SIZE + col) + (block_row * BLOCK_SIZE + row) * D];
        }
        else
        {
            a_block[row][col] = 0.0;
        }

        if (block_col * BLOCK_SIZE + col < N && m * BLOCK_SIZE + row < D)
        {
            b_block[row][col] = b[(block_col * BLOCK_SIZE + col) * D + (m * BLOCK_SIZE + row)];
        }
        else
        {
            b_block[row][col] = 0.0;
        }

        __syncthreads();

        #pragma unroll
        for (int e = 0; e < BLOCK_SIZE; e++)
        {
            c_value += fabsf(a_block[row][e] - b_block[e][col]);
        }

        __syncthreads();
    }

    if (block_col * BLOCK_SIZE + col < N && block_row * BLOCK_SIZE + row < M)
    {
        c[(block_col * BLOCK_SIZE + col) + (block_row * BLOCK_SIZE + row) * N] = c_value;
    } 
}

void cuda_l1_t(float *c, const float *a, const float *b, const int M, const int N, const int D)
{
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

    l1_t_kernel<<<dimGrid, dimBlock>>>(c, a, b, M, N, D);
    cudaDeviceSynchronize();
}