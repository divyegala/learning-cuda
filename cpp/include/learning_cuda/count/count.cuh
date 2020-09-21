#include "count_kernels.cuh"

namespace naive {

template <typename T, class F>
int count_if(T* arr, int size,
             F count_if_op) {
    int *count_d;
    int *count_h = new int[1];
    cudaMalloc(&count_d, sizeof(int));
    cudaMemset(count_d, 0, sizeof(int));

    int TPB = std::ceil((float) size / N_BLK);
    
    count_kernel<<<TPB, N_BLK>>> (arr, size, count_d, count_if_op);

    cudaMemcpy(count_h, count_d, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(count_d);

    return *count_h;
}

} // namespace naive
    