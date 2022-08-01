#include <mpi.h>
#include <omp.h>
#include <iostream>
#include <string>

int GetThreadCount( int argc, char** argv );
int GetThreadCount( int argc, char** argv )
{
    int nThreads = 1;
    if ( argc < 2 )
    {
        return nThreads;
    }
    std::size_t pos{};
    nThreads = std::stoi( argv[1], &pos );
    return nThreads;
}

int main(int argc, char *argv[])
{
    int nThreads = GetThreadCount( argc, argv );
    //std::printf(" nThreads = %d\n", nThreads);
    omp_set_num_threads( nThreads );

    int numprocs, rank, namelen;
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    //int iam = 0, np = 1;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Get_processor_name(processor_name, &namelen);

    #pragma omp parallel
    {
        int np = omp_get_num_threads();
        int iam = omp_get_thread_num();
        std::printf("Hybrid: Hello from thread %d out of %d from process %d out of %d on %s\n",
                    iam, np, rank, numprocs, processor_name);
    }

    MPI_Finalize();

    return 0;
}
