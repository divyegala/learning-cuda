#pragma once

#include "utils.cuh"

namespace naive {
namespace detail {

template <typename T, typename NormOp>
__global__
void norm_kernel(T *data, T *indptr, int m, T *row_sums, NormOp norm_op) {
    // one thread per row

    int row = threadIdx.x + blockIdx.x * blockDim.x;

    if (row < m) {
        int row_start = indptr[row];
        int row_end = indptr[row + 1];

        T sum = 0;
        for(int element = row_start; element < row_end; element++) {
            sum += norm_op(data[element]);
        }
        // row_sums[row] = sum;

        if (sum == 0)
            return;
        
        for(int element = row_start; element < row_end; element++) {
            data[element] /= sum;
        }
    }
}

} // namespace detail
} // namespace naive

#define FULL_MASK 0xffffffff

namespace warp {
namespace detail {

template <typename T, int TPB, typename NormOp>
__global__
void norm_kernel(T *data, T *indptr, int m, T *row_sums, NormOp norm_op) {

    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    int row = tid / 32;
    int lane_id = tid % 32;

    // one warp per row
    T sum = 0;

    if (row < m) {
        int row_start = indptr[row];
        int row_end = indptr[row + 1];

        for(int element = row_start + lane_id; element < row_end; element += 32) {
            sum += norm_op(data[element]);
        }
    }

    // reduce across warps (rows)
    for(int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(FULL_MASK, sum, offset);
    }

    __shared__ int row_sum;

    // Assuming block size 32
    if (lane_id == 0) {
        row_sum = sum;
    }

    if (row < m && sum != 0) {
        int row_start = indptr[row];
        int row_end = indptr[row + 1];

        for(int element = row_start + lane_id; element < row_end; element += 32) {
            data[element] /= row_sum;
        }
    }
}

} // namespace detail
} // namespace warp