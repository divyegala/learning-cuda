#include <cuda_runtime.h>

#define N_BLK 32

namespace naive {

template <typename T, class F>
__global__ 
void count_kernel(T* arr, int size, int *count, F count_if_op) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < size) {
        if (count_if_op(arr[tid])) {
            atomicAdd(count, 1);
        }
    }
}

} // namespace naive