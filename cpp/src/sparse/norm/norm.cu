#include <iostream>
#include <cstdlib>
#include <functional>
#include <cmath>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <learning_cuda/sparse/norm/norm.cuh>

template <typename T>
csr_matrix_t<T>* _random_device_csr(int m, int n, float density) {
    csr_matrix_t<T> *csr_d;
    csr_d = (csr_matrix_t<T> *) malloc(sizeof(csr_matrix_t<T>));
    csr_d->m = m;
    csr_d->n = n;

    thrust::host_vector<T> data_h, indices_h, indptr_h;
    indptr_h.push_back(0);

    int nnz = 0;
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < n; j++) {
            if (( (float) std::rand() / RAND_MAX) < density) {
                T element = std::rand() % 25;
                data_h.push_back(element);
                indices_h.push_back(j);
                // std::cout << element << " ";
                nnz++;
            }
            else {
                // std::cout << 0 << " ";
            }
        }
        indptr_h.push_back(nnz);
        // std::cout << std::endl;
    }

    csr_d->nnz = nnz;
    csr_d->data.resize(nnz);
    csr_d->indices.resize(nnz);
    csr_d->indptr.resize(m + 1);
    
    thrust::copy(data_h.begin(), data_h.end(), csr_d->data.begin());
    thrust::copy(indices_h.begin(), indices_h.end(), csr_d->indices.begin());
    thrust::copy(indptr_h.begin(), indptr_h.end(), csr_d->indptr.begin());

    return csr_d;
}

int main() {

    csr_matrix_t<int> *csr_d = _random_device_csr<int>(2000, 500000, 0.1);

    // std::cout << "CSR data: ";
    // thrust::copy(csr_d->data.begin(), csr_d->data.end(), std::ostream_iterator<int>(std::cout, " "));
    // std::cout << std::endl;

    // std::cout << "CSR indices: ";
    // thrust::copy(csr_d-> indices.begin(), csr_d->indices.end(), std::ostream_iterator<int>(std::cout, " "));
    // std::cout << std::endl;

    // std::cout << "CSR indptr: ";
    // thrust::copy(csr_d->indptr.begin(), csr_d->indptr.end(), std::ostream_iterator<int>(std::cout, " "));
    // std::cout << std::endl;

    cudaDeviceSynchronize();
    naive::norm<int, 32>(*csr_d, L1Op<int>());

    cudaDeviceSynchronize();
    warp::norm<int, 32>(*csr_d, L1Op<int>());

    delete csr_d;

    return 0;
}