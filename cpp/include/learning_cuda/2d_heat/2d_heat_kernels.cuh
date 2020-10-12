#pragma once

namespace naive {
namespace detail {

template <typename T, int TPB>
__global__
void heat_kernel(const T * __restrict__ data_d_old, T * __restrict__ data_d_new,
                 int nx, int ny) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    int self_idx = i * ny + j;

    int top_idx = (i - 1) * ny + j;
    int bottom_idx = (i + 1) * ny + j;
    int left_idx = i * ny + (j - 1);
    int right_idx = i * ny + (j + 1);

    if ((i > 0 && i < nx - 1) && (j > 0 && j < ny - 1)) {
        // global mem, not updating any boundary points
        data_d_new[self_idx] = 0.25 * (data_d_old[top_idx] + data_d_old[bottom_idx] +
                                       data_d_old[left_idx] + data_d_old[right_idx]);
    }

}

} // namespace detail
} // namespace naive

namespace shared_global {
namespace detail {

template <typename T, int TPB>
__global__
void heat_kernel(const T * __restrict__ data_d_old, T * __restrict__ data_d_new,
                 int nx, int ny) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    __shared__ T data_shared[TPB][TPB];

    int self_idx = i * ny + j;

    if (i < nx && j < ny) {
        data_shared[threadIdx.x][threadIdx.y] = data_d_old[self_idx];
    }
    __syncthreads();

    int top_idx = (i - 1) * ny + j;
    int bottom_idx = (i + 1) * ny + j;
    int left_idx = i * ny + (j - 1);
    int right_idx = i * ny + (j + 1);

    if ((i > 0 && i < nx - 1) && (j > 0 && j < ny - 1)) {
        if ((threadIdx.x > 0 && threadIdx.x < TPB - 1) && (threadIdx.y > 0 && threadIdx.y < TPB - 1))
        {
            // pick from shared memory
            data_d_new[self_idx] = 0.25 * (data_shared[threadIdx.x - 1][threadIdx.y] + 
                                           data_shared[threadIdx.x + 1][threadIdx.y] +
                                           data_shared[threadIdx.x][threadIdx.y - 1] + 
                                           data_shared[threadIdx.x][threadIdx.y + 1]);
        }
        else {
            // global mem, not updating any boundary points
            data_d_new[self_idx] = 0.25 * (data_d_old[top_idx] + data_d_old[bottom_idx] +
                                           data_d_old[left_idx] + data_d_old[right_idx]);
        }
    }

}

} // namespace detail
} // namespace shared_global

namespace shared_only {
namespace detail {
    
template <typename T, int TPB>
__global__
void heat_kernel(const T * __restrict__ data_d_old, T * __restrict__ data_d_new,
                    int nx, int ny) {
    int i = threadIdx.x + (TPB - 2) * blockIdx.x;
    int j = threadIdx.y + (TPB - 2) * blockIdx.y;

    __shared__ T data_shared[TPB][TPB];

    int self_idx = i * ny + j;

    if (i < nx && j < ny) {
        data_shared[threadIdx.x][threadIdx.y] = data_d_old[self_idx];
    }
    __syncthreads();

    if ((i > 0 && i < nx - 1) && (j > 0 && j < ny - 1)) {
        if ((threadIdx.x > 0 && threadIdx.x < TPB - 1) && (threadIdx.y > 0 && threadIdx.y < TPB - 1))
        {
            // pick from shared memory
            data_d_new[self_idx] = 0.25 * (data_shared[threadIdx.x - 1][threadIdx.y] + 
                                           data_shared[threadIdx.x + 1][threadIdx.y] +
                                           data_shared[threadIdx.x][threadIdx.y - 1] + 
                                           data_shared[threadIdx.x][threadIdx.y + 1]);
        }
    }

}

} // namespace detail
} // namespace shared_only