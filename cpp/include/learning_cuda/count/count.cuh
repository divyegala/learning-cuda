#include "count_kernels.cuh"

#define TPB 32

namespace common {

void _prep_count_if(int **count_d, int &N_BLK, int size) {
    cudaMalloc(count_d, sizeof(int));
    cudaMemset(*count_d, 0, sizeof(int));

    N_BLK = std::ceil((float) size / TPB);
}

void _finish_count_if(int **count_d, int &count_h) {
    cudaMemcpy(&count_h, *count_d, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(*count_d);
}

} // namespace common

namespace naive {

template <typename T, class F>
int count_if(T* arr, int size,
             F count_if_op) {
    int *count_d, count_h, N_BLK;

    common::_prep_count_if(&count_d, N_BLK, size);

    count_kernel<<<N_BLK, TPB>>> (arr, size, count_d, count_if_op);

    common::_finish_count_if(&count_d, count_h);

    return count_h;
}

} // namespace naive

namespace manual_reduction {

template <typename T, class F>
int count_if(T* arr, int size,
             F count_if_op) {
    int *count_d, count_h, N_BLK;

    common::_prep_count_if(&count_d, N_BLK, size);

    count_kernel<<<N_BLK, TPB>>> (arr, size, count_d, count_if_op);

    common::_finish_count_if(&count_d, count_h);

    return count_h;
}

} // namespace manual_reduction