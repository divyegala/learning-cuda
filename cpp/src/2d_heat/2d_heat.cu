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

    int nx = 32, ny = 5;
    int NBLK = std::ceil((float) nx / 32);

    thrust::device_vector<double> data_d_old(nx * ny, 0);

    _initialize_temps<<<NBLK, 32>>> (thrust::raw_pointer_cast(data_d_old.data()),
                                     nx, ny);

    thrust::device_vector<double> data_d_new(nx * ny, 0);

    _initialize_temps<<<NBLK, 32>>> (thrust::raw_pointer_cast(data_d_new.data()),
                                    nx, ny);

    int iter = 1;
    double *data_old_ptr = thrust::raw_pointer_cast(data_d_old.data());
    double *data_new_ptr = thrust::raw_pointer_cast(data_d_new.data());
    naive::heat_diffusion<double, 16>(data_old_ptr,
                                      data_new_ptr,
                                      nx, ny, iter);

    // Printing device vector
    thrust::copy(data_d_new.begin(), data_d_new.end(),
                 std::ostream_iterator<int>(std::cout, " "));

    return 0;
}