#include "SolverDetailCpu.h"
#include "Cmpi.h"
#include <cstdio>

void SolverInitCpu()
{
    Cmpi::num_gpus = 0;
    std::printf("no CUDA capable devices were detected\n");
}

void CfdCopyVectorCpu( float * a, float * b, int ni )
{
    for ( int i = 0; i < ni; ++ i )
    {
        a[ i ] = b[ i ];
    }
}

void CfdScalarUpdateCpu( float * q, float * qn, float c, float * timestep, float * ds, int ni )
{
    for ( int i = 1; i < ni + 1; ++ i )
    {
        float cfl = c * timestep[ i ] / ds[ i ];
        q[ i ] = qn[ i ] - cfl * ( qn[ i ] - qn[ i - 1 ] );
    }
}

