#include "Cmpi.h"

int Cmpi::pid = 0;
int Cmpi::nproc = 1;
int Cmpi::serverid = 0;
int Cmpi::num_gpus = 0;
int Cmpi::num_cpus = 1;

Cmpi::Cmpi()
{
    ;
}

Cmpi::~Cmpi()
{
    ;
}

void Cmpi::Init(int argc, char **argv)
{
#ifdef PRJ_ENABLE_MPI
    MPI_Init( &argc, &argv ); 
    MPI_Comm_rank( MPI_COMM_WORLD, &Cmpi::pid ); 
    MPI_Comm_size( MPI_COMM_WORLD, &Cmpi::nproc );
#endif
}

void Cmpi::Finalize()
{
#ifdef PRJ_ENABLE_MPI
    MPI_Finalize();
#endif
}