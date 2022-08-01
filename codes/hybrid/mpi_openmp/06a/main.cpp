#include <mpi.h>
#include <omp.h>
#include <iostream>

int main(int argc, char *argv[])
{
    int my_id, omp_rank;
    int provided;
    //int required=MPI_THREAD_FUNNELED;
    int required=MPI_THREAD_MULTIPLE;
    
    MPI_Init_thread(&argc, &argv, required, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
    int nthreads = omp_get_num_threads();
    std::printf("OpenMP: Number of threads = %d\n", nthreads);
    #pragma omp parallel num_threads(4) private(omp_rank)
    {
        //omp_rank = omp_get_thread_num();
		omp_rank = omp_get_thread_num();
		int local_nthreads = omp_get_num_threads();
		std::printf("OpenMP: Number of local_nthreads = %d\n", local_nthreads);
        std::printf("I'm thread %d in process %d\n", omp_rank, my_id);
    }
    MPI_Finalize();

    return 0;
}
