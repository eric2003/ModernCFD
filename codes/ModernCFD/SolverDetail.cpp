#include "SolverDetail.h"
#include "SolverDetailCpu.h"
#ifdef PRJ_ENABLE_CUDA
#include "SolverDetailCuda.h"
#endif

void SolverInit()
{
#ifdef PRJ_ENABLE_CUDA
    SolverInitCuda();
#else
    SolverInitCpu();
#endif
}

void SetDevice( int cpu_thread_id )
{
#ifdef PRJ_ENABLE_CUDA
    SetDeviceCuda( cpu_thread_id );
#else

#endif
}

void CfdCopyVector( float * a, float * b, int ni )
{
#ifdef PRJ_ENABLE_CUDA
    CfdCopyVectorCuda( a,  b, ni );
#else
    CfdCopyVectorCpu( a,  b, ni );
#endif
}

void CfdScalarUpdate( float * q, float * qn, float c, float * timestep, float * ds, int ni )
{
#ifdef PRJ_ENABLE_CUDA
    CfdScalarUpdateCuda( q, qn, c, timestep, ds, ni );
#else
    CfdScalarUpdateCuda( q, qn, c, timestep, ds, ni );
#endif

}