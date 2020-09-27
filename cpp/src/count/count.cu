#include <iostream>
#include <cstdlib>
#include <functional>
#include <cmath>

#include <thrust/execution_policy.h>
#include <thrust/count.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <learning_cuda/count/count.cuh>


int main() {
    int n_elems = 10000000;

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

    cudaDeviceSynchronize();
    int thrust_count = thrust::count_if(thrust::device, rand_d.begin(),
                                        rand_d.end(), is_greater_than_10);
    
    std::cout << "\nThrust Count: " << thrust_count << std::endl;
    
    int *rand_d_ptr = thrust::raw_pointer_cast(rand_d.data());

    cudaDeviceSynchronize();
    int naive_count = naive::count_if<int, 64>(rand_d_ptr, n_elems,
                                                is_greater_than_10);

    std::cout << "\nNaive Count: " << naive_count << std::endl;

    cudaDeviceSynchronize();
    int man_count = manual_reduction::count_if<int, 64>(rand_d_ptr, n_elems,
                                                         is_greater_than_10);

    std::cout << "\nManual Reduction Count: " << man_count << std::endl;

    cudaDeviceSynchronize();
    int syn_count = syncthreads_count_reduction::count_if<int, 64>(rand_d_ptr,
                                                                   n_elems,
                                                                   is_greater_than_10);

    std::cout << "\nSyncthread Reduction Count: " << syn_count << std::endl;

    cudaDeviceSynchronize();
    int bal_count = ballot_sync_reduction::count_if<int, 64>(rand_d_ptr, 
                                                             n_elems,
                                                             is_greater_than_10);

    std::cout << "\nBallot Reduction Count: " << bal_count << std::endl;

    return 0;
}