#pragma once

template <typename T>
struct csr_matrix_t {
    thrust::device_vector<T> data, indices, indptr;
    // T *data, *indices, *indptr;
    int m, n, nnz = 0;
};

template <typename T>
struct L1Op {
    __device__
    T operator()(const T x) {
        return x > 0 ? x : -x;
    }
};