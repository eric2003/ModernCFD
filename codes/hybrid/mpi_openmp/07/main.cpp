#include <cstdio>
#include <cstdlib>
#include <string>
#include <omp.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int provided;
    int rank;

    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided != MPI_THREAD_FUNNELED) {
        fprintf(stderr, "Warning MPI did not provide MPI_THREAD_FUNNELED\n");
    }

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    #pragma omp parallel default(none), \
                        shared(rank), \
                        shared(ompi_mpi_comm_world), \
                        shared(ompi_mpi_int), \
                        shared(ompi_mpi_char)
    {
        printf("Hello from thread %d at rank %d parallel region\n", 
                omp_get_thread_num(), rank);
        #pragma omp master
        {
            char helloWorld[12];
            if (rank == 0) {
                strcpy(helloWorld, "Hello World");
                MPI_Send(helloWorld, 12, MPI_CHAR, 1, 0, MPI_COMM_WORLD);
                std::printf("Rank %d send: %s\n", rank, helloWorld);
            }
            else {
                MPI_Recv(helloWorld, 12, MPI_CHAR, 0, 0, MPI_COMM_WORLD,
                         MPI_STATUS_IGNORE);
                std::printf("Rank %d received: %s\n", rank, helloWorld);
            }
        }

    }

    MPI_Finalize();
    return 0;
}
