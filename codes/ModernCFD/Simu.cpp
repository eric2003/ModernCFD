#include "Simu.h"
#include "Geom.h"
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
    Geom * geom = new Geom{};
    geom->Init();
    geom->GenerateGrid();

    //cfd parameter
    CfdPara * cfd_para = new CfdPara{};
    cfd_para->Init( geom );

    Solver * solver = new Solver{};
    solver->Run( cfd_para, geom );
    delete cfd_para;
    delete geom;
    delete solver;
}