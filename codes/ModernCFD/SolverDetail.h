#pragma once

void SolverInit();
void SetDevice( int cpu_thread_id );
void CfdCopyVector( float * a, float * b, int ni );
void CfdScalarUpdate( float * q, float * qn, float c, float * timestep, float * ds, int ni );
