#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
 
/**
 * @brief Illustrates how to initialise the MPI environment with multithreading
 * support and ask for the MPI_THREAD_MULTIPLE level.
 * @details This application initialised MPI and asks for the 
 * MPI_THREAD_MULTIPLE thread support level. It then compares it with the
 * thread support level provided by the MPI implementation.
 **/
int main(int argc, char* argv[])
{
    // Initilialise MPI and ask for thread support
    int provided;
    MPI_Init_thread(NULL, NULL, MPI_THREAD_MULTIPLE, &provided);
    if(provided < MPI_THREAD_MULTIPLE)
    {
        printf("The threading support level is lesser than that demanded.\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    else
    {
        printf("The threading support level corresponds to that demanded.\n");
    }
 
    // Tell MPI to shut down.
    MPI_Finalize();
 
    return EXIT_SUCCESS;
}
