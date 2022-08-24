#include "Solver.h"
#include <string>
#include <set>
#include <map>
#include <mpi.h>
#include "Cmpi.h"
#include "Grid.h"

Solver::Solver()
{
}

Solver::~Solver()
{
    ;
}

void Solver::Init()
{
}

void Solver::Run( CfdPara * cfd_para, Geom * geom, BoundarySolver * bcSolver, int zoneId )
{
    CfdSolve( geom->ni_global, geom->xcoor_global, cfd_para->simu_time, zoneId, geom->ni, cfd_para->nt, cfd_para->cfl, geom->xcoor, bcSolver );
}
