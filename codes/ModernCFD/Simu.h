#pragma once
#include <mpi.h>

class Simu
{
public:
    Simu( int argc, char ** argv );
    ~Simu();
public:
    void Init( int argc, char ** argv );
    void Run();
};

