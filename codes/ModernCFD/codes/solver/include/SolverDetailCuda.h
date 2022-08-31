#pragma once
#ifdef PRJ_ENABLE_CUDA

void SolverInitCuda();
void SetDeviceCuda( int cpu_thread_id );
void CfdCopyVectorCuda( float * a, float * b, int ni );
void CfdScalarUpdateCuda( float * q, float * qn, float c, float * timestep, float * ds, int ni );
#endif