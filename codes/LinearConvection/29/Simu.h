#pragma once
#include <mpi.h>

class Simu
{
public:
    Simu();
    ~Simu();
public:
    void Init(int argc, char **argv);
    void Run();
};

