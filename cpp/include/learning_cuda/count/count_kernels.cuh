#pragma once

#include <cuda_runtime.h>

namespace naive {
namespace detail {

template <typename T, typename CountIfOp>
__global__ 
void count_kernel(T* arr, int size, int *count, CountIfOp count_if_op) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < size) {

        if (count_if_op(arr[tid])) {
            atomicAdd(count, 1);
        }
    }
}

} // namespace detail
} // namespace naive

namespace manual_reduction {
namespace detail {

template <typename T, int TPB, typename CountIfOp>
__global__ 
void count_kernel(T* arr, int size, int *count, CountIfOp count_if_op) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    __shared__ unsigned int local_count_array[TPB];

    if (tid < size) {
        if (count_if_op(arr[tid])) {
            local_count_array[threadIdx.x] = 1;
        }
        else {
            local_count_array[threadIdx.x] = 0;
        }

        __syncthreads();

        for (int offset = blockDim.x / 2; offset > 0; offset >>=1 ) {
            if (threadIdx.x < offset && tid + offset < size) {
                local_count_array[threadIdx.x] += local_count_array[threadIdx.x + offset];
                __syncthreads();
            }
        }

        if(threadIdx.x == 0) {
            atomicAdd(count, local_count_array[threadIdx.x]);
        }
    }
}

} // namespace detail
} // namespace manual_reduction

namespace syncthreads_count_reduction {
namespace detail {

template <typename T, typename CountIfOp>
__global__ 
void count_kernel(T* arr, int size, int *count, CountIfOp count_if_op) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    bool predicate = tid < size && count_if_op(arr[tid]);

    int block_count = __syncthreads_count(predicate);

    if(threadIdx.x == 0) {
        atomicAdd(count, block_count);
    }
    
}

} // namespace detail
} // namespace syncthreads_count

#define FULL_MASK 0xffffffff

namespace ballot_sync_reduction {
namespace detail {

template <typename T, int TPB, typename CountIfOp>
__global__ 
void count_kernel(T* arr, int size, int *count, CountIfOp count_if_op) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    bool predicate = tid < size && count_if_op(arr[tid]);

    unsigned ballot_mask = __ballot_sync(FULL_MASK, predicate);
    int warp_count = __popc(ballot_mask);

    if(threadIdx.x % 32 == 0) {
        atomicAdd(count, warp_count);
    }
    
}

} // namespace detail
} // namespace ballot_sync_reduction