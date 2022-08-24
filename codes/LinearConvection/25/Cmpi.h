#pragma once
#include <mpi.h>

class Cmpi
{
public:
    Cmpi();
    ~Cmpi();
public:
    static void Init(int argc, char **argv);
public:
    static int pid;
    static int nproc;
    static int serverid;
};

