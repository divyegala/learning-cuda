#include <iostream>
#include <cstdlib>
#include <functional>
#include <cmath>

#include <thrust/execution_policy.h>
#include <thrust/count.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "count_impl.cuh"


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