#include "Simu.h"
#include "Grid.h"
#include "Cmpi.h"

Simu::Simu()
{
    ;
}

Simu::~Simu()
{
    ;
}

void Simu::Init(int argc, char **argv)
{
}

void Simu::Run()
{
    //cfd parameter
    CfdPara * cfd_para = new CfdPara{};
    cfd_para->Init();

    int nZones = Cmpi::nproc;
    Geom * geom = new Geom();
    geom->Init( nZones );
    int zoneId = Cmpi::pid;

    BoundarySolver bcSolver;
    bcSolver.zoneId = zoneId;
    bcSolver.Init( zoneId, nZones, geom->ni );
    geom->GenerateGrid( cfd_para );
    CfdSolve( geom->ni_global, geom->xcoor_global, cfd_para->simu_time, zoneId, geom->ni, cfd_para->nt, cfd_para->cfl, geom->xcoor, &bcSolver );
    delete cfd_para;
    delete geom;
}