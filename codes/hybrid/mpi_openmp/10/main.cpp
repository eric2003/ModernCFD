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
	    std::printf("my_nthreads= %d\n", my_nthreads);
	}
	#pragma omp parallel num_threads(10)
	{
		int my_nthreads1 = omp_get_num_threads();
	    std::printf("my_nthreads1= %d\n", my_nthreads1);
	}	
	return 0;	
}
