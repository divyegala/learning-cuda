#pragma once

#include <cuda_runtime.h>

#include "2d_heat_kernels.cuh"

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

namespace shared_global {

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

} // namespace shared_global

namespace shared_only {

template<typename T, int TPB>
void heat_diffusion(T *data_d_old, T *data_d_new, int nx, int ny, int iter) {
    dim3 block_size(TPB, TPB);
    dim3 grid_size(std::ceil((float) nx / (TPB - 2)),
                    std::ceil((float) ny / (TPB - 2)));
    for(int i = 0; i < iter; i += 2) {
        detail::heat_kernel<T, TPB><<<grid_size, block_size>>> (data_d_new,
                                                                data_d_old,
                                                                nx, ny);
        detail::heat_kernel<T, TPB><<<grid_size, block_size>>> (data_d_old,
                                                                data_d_new,
                                                                nx, ny);
    }

}

} // namespace shared_only