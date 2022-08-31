#pragma once
#ifdef PRJ_ENABLE_MPI
#include <mpi.h>
#endif
class Cmpi
{
public:
    Cmpi();
    ~Cmpi();
public:
    static void Init(int argc, char **argv);
    static void Finalize();
public:
    static int pid;
    static int nproc;
    static int serverid;
    static int num_gpus;
    static int num_cpus;
};

