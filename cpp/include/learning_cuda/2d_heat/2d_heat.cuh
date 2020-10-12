#pragma once

#include <iostream>

#include <cuda_runtime.h>

#include "2d_heat_kernels.cuh"

#include <thrust/device_vector.h>

namespace naive {

template<typename T, int TPB>
void heat_diffusion(T *data_d_old, T *data_d_new, int nx, int ny, int iter) {
    dim3 block_size(TPB, TPB);
    dim3 grid_size(std::ceil((float) nx / TPB),
                   std::ceil((float) ny / TPB));
    for(int i = 0; i < iter; i += 2) {
        detail::heat_kernel<T, TPB><<<grid_size, block_size>>> (data_d_new,
                                                                data_d_old,
                                                                nx, ny);

        detail::heat_kernel<T, TPB><<<grid_size, block_size>>> (data_d_old,
                                                                data_d_new,
                                                                nx, ny);
    }


}

} // namespace naive