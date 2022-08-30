#pragma once

void SolverInitCpu();
void CfdCopyVectorCpu( float * a, float * b, int ni );
void CfdScalarUpdateCpu( float * q, float * qn, float c, float * timestep, float * ds, int ni );