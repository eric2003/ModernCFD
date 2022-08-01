#include <cstdio>
#include <cstdlib>
#include <string>
#include <omp.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
	omp_set_num_threads(4);
	int nthreads = omp_get_num_threads();
	std::printf("nthreads= %d\n", nthreads);
    #pragma omp parallel
    {
        int my_nthreads = omp_get_num_threads();
        int rank = omp_get_thread_num();
        std::printf("rank = %d, my_nthreads= %d\n", rank, my_nthreads);
    }
	omp_set_num_threads(2);
	int nthreads_1 = omp_get_num_threads();
    std::printf("nthreads_1= %d\n", nthreads_1);	
    #pragma omp parallel
    {
        int my_nthreads_1 = omp_get_num_threads();
        int rank = omp_get_thread_num();
        std::printf("rank = %d, my_nthreads_1= %d\n", rank, my_nthreads_1);
    }
}
