#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <learning_cuda/2d_heat/2d_heat.cuh>

template <typename T>
__global__
void _initialize_temps(T *data_d, int nx, int ny) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // left wall is 1
    // row major format
    if (tid * ny < nx * ny)
        data_d[tid * ny] = 1;
}

int main() {

    int nx = 2000, ny = 2000;
    int NBLK = std::ceil((float) nx / 32);
    int iter = 100;

    // naive
    thrust::device_vector<float> data_d_old(nx * ny, 0);

    _initialize_temps<<<NBLK, 32>>> (thrust::raw_pointer_cast(data_d_old.data()),
                                     nx, ny);

    thrust::device_vector<float> data_d_new = data_d_old;

    cudaDeviceSynchronize();
    naive::heat_diffusion<float, 16>(thrust::raw_pointer_cast(data_d_old.data()),
                                      thrust::raw_pointer_cast(data_d_new.data()),
                                      nx, ny, iter);
    cudaDeviceSynchronize();
    // Printing device vector
    // std::cout << "\n Naive: \n";
    // thrust::copy(data_d_new.begin(), data_d_new.end(),
    //              std::ostream_iterator<double>(std::cout, " "));

    // shared_global
    thrust::device_vector<float> data_d_old_sg(nx * ny, 0);

    _initialize_temps<<<NBLK, 32>>> (thrust::raw_pointer_cast(data_d_old_sg.data()),
                                     nx, ny);

    thrust::device_vector<float> data_d_new_sg = data_d_old_sg;

    cudaDeviceSynchronize();
    shared_global::heat_diffusion<float, 16>(thrust::raw_pointer_cast(data_d_old_sg.data()),
                                              thrust::raw_pointer_cast(data_d_new_sg.data()),
                                              nx, ny, iter);
    cudaDeviceSynchronize();
    // Printing device vector
    // std::cout << "\n Shared & Global: \n";
    // thrust::copy(data_d_new_sg.begin(), data_d_new_sg.end(),
    //              std::ostream_iterator<double>(std::cout, " "));

    // shared_only
    thrust::device_vector<float> data_d_old_so(nx * ny, 0);

    _initialize_temps<<<NBLK, 32>>> (thrust::raw_pointer_cast(data_d_old_so.data()),
                                     nx, ny);

    thrust::device_vector<float> data_d_new_so = data_d_old_so;

    cudaDeviceSynchronize();
    shared_only::heat_diffusion<float, 16>(thrust::raw_pointer_cast(data_d_old_so.data()),
                                              thrust::raw_pointer_cast(data_d_new_so.data()),
                                              nx, ny, iter);
    cudaDeviceSynchronize();
    // Printing device vector
    // std::cout << "\n Shared Only: \n";
    // thrust::copy(data_d_new_so.begin(), data_d_new_so.end(),
    //              std::ostream_iterator<double>(std::cout, " "));

    return 0;
}