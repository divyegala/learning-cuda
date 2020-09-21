#include <iostream>
#include <cstdlib>
#include <functional>
#include <cmath>

#include <thrust/execution_policy.h>
#include <thrust/count.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <learning_cuda/count/count_kernels.cuh>

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

    return *count_h;
}

} // namespace naive

int main() {
    int n_elems = 100000;

    thrust::device_vector<int> rand_d(n_elems, 0);

    for(int i = 0; i < n_elems; i++) {
        rand_d[i] = std::rand() % 25;
    }

    // Printing device vector
    // thrust::copy(rand_d.begin(), rand_d.end(),
    //              std::ostream_iterator<int>(std::cout, " "));

    auto is_greater_than_10 = [] __device__ (const int x) {
        return (x > 10 ? true : false);
    };

    int thrust_count = thrust::count_if(thrust::device, rand_d.begin(),
                                        rand_d.end(), is_greater_than_10);
    
    std::cout << "\nThrust Count: " << thrust_count << std::endl;
    
    int naive_count = naive::count_if(thrust::raw_pointer_cast(rand_d.data()),
                                      n_elems, is_greater_than_10);

    std::cout << "\nNaive Count: " << naive_count << std::endl;

    return 0;
}