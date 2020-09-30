#pragma once

#include "utils.cuh"

namespace naive {
namespace detail {

template <typename T, typename NormOp>
__global__
void sum_kernel(T *data, T *indptr, int m, T *row_sums, NormOp norm_op) {
    // one thread per row

    int row = threadIdx.x + blockIdx.x * blockDim.x;

    if (row < m) {
        int row_start = indptr[row];
        int row_end = indptr[row + 1];

        T sum = 0;
        for(int element = row_start; element < row_end; element++) {
            sum += norm_op(data[element]);
        }
        row_sums[row] = sum;
    }
}

} // namespace detail
} // namespace naive

#define FULL_MASK 0xffffffff

namespace warp {
namespace detail {

template <typename T, typename NormOp>
__global__
void sum_kernel(T *data, T *indptr, int m, T *row_sums, NormOp norm_op) {
    // one thread per row

    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    int warp_id = tid / 32;
    int lane_id = tid % 32;

    int row = warp_id;

    // one warp per row
    T sum = 0;

    if (row < m) {
        int row_start = indptr[row];
        int row_end = indptr[row + 1];

        for(int element = row_start + lane_id; element < row_end; element += 32) {
            sum += norm_op(data[element]);
        }
    }

    for(int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(FULL_MASK, sum, offset);
    }

    // Assuming block size 32
    if (row < m && lane_id == 0) {
        row_sums[row] = sum;
    }
}

} // namespace detail
} // namespace warp