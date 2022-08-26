#include "Simu.h"
#include "Geom.h"
#include "Solver.h"
#include "Cmpi.h"
#include "CfdPara.h"

Simu::Simu(int argc, char **argv)
{
    Cmpi::Init( argc, argv );
}

Simu::~Simu()
{
    Cmpi::Finalize();
}

void Simu::Init(int argc, char **argv)
{
}

void Simu::Run()
{
    Geom_t::Init();
    Geom * geom = new Geom{};
    geom->Init();
    geom->GenerateGrid();
    geom->ComputeGeom();

    //cfd parameter
    CfdPara * cfd_para = new CfdPara{};
    cfd_para->Init( geom );

    Solver * solver = new Solver{};
    solver->Run( cfd_para, geom );
    delete cfd_para;
    delete geom;
    delete solver;
}