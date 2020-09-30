#pragma once

#include "norm_kernels.cuh"

namespace naive {

template <typename T, int TPB, typename NormOp>
void norm(csr_matrix_t<T>& csr_d, NormOp norm_op) {
    thrust::device_vector<T> row_sums_d (csr_d.m, 0);

    int N_BLK = std::ceil((float) csr_d.m / TPB);

    detail::sum_kernel<T> <<<N_BLK, TPB>>> (thrust::raw_pointer_cast(csr_d.data.data()),
                                            thrust::raw_pointer_cast(csr_d.indptr.data()),
                                            csr_d.m,
                                            thrust::raw_pointer_cast(row_sums_d.data()),
                                            norm_op);

    // Printing device vector
    // std::cout << "naive row sums: ";
    // thrust::copy(row_sums_d.begin(), row_sums_d.end(),
    //              std::ostream_iterator<int>(std::cout, " "));
    // std::cout << std::endl;
}

} // namespace naive

namespace warp {

template <typename T, int TPB, typename NormOp>
void norm(csr_matrix_t<T>& csr_d, NormOp norm_op) {
    thrust::device_vector<T> row_sums_d (csr_d.m, 0);

    int N_BLK = csr_d.m;

    detail::sum_kernel<T> <<<N_BLK, TPB>>> (thrust::raw_pointer_cast(csr_d.data.data()),
                                            thrust::raw_pointer_cast(csr_d.indptr.data()),
                                            csr_d.m,
                                            thrust::raw_pointer_cast(row_sums_d.data()),
                                            norm_op);

    // Printing device vector
    // std::cout << "warp row sums: ";
    // thrust::copy(row_sums_d.begin(), row_sums_d.end(),
    //                 std::ostream_iterator<int>(std::cout, " "));
    // std::cout << std::endl;
}

} // namespace warp