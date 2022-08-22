#pragma once

void GenerateGrid( int ni, float xmin, float xmax, float * xcoor );
void CfdSimulation( int ni, float * xcoor );
void CfdSolve( int zoneId, int ni, int nt, float cfl, float * xcoor );