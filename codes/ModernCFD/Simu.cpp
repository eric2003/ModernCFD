#include "Simu.h"
#include "Grid.h"
#include "Solver.h"
#include "Cmpi.h"
#include "CfdPara.h"

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
    Geom * geom = new Geom{};
    geom->Init();
    geom->GenerateGrid( cfd_para );

    Solver * solver = new Solver{};
    solver->Run( cfd_para, geom );
    delete cfd_para;
    delete geom;
    delete solver;
}