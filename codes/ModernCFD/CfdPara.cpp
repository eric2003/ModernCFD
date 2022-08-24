#include "CfdPara.h"

CfdPara::CfdPara()
{
    //this->Init();
}

CfdPara::~CfdPara()
{
    ;
}

void CfdPara::Init()
{
    int ni_global = 41;
    this->xmin = 0.0;
    this->xmax = 2.0;

    this->cfl = 0.5;
    this->simu_time = 0.625;
    this->xlen = xmax - xmin;
    this->dx = xlen / ( ni_global - 1 );
    this->cspeed = 1.0;
    this->dt = dx * cfl / cspeed;
    this->fnt = ( simu_time + SMALL ) / dt;
    this->nt = fnt;
}
