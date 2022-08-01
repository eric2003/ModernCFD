#include <cstdio>
#include <cstdlib>
#include <string>
#include <omp.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
	omp_set_num_threads(4);
	int nthreads = omp_get_num_threads();
	std::printf("nthreads= %d\n", nthreads);
}
