#include "CfdPara.h"
#include "Geom.h"

CfdPara::CfdPara()
{
    //this->Init();
}

CfdPara::~CfdPara()
{
    ;
}

void CfdPara::Init( Geom * geom )
{
    int ni_global = 41;

    this->cfl = 0.5;
    this->simu_time = 0.625;
    this->cspeed = 1.0;
    this->dt = geom->dx * cfl / cspeed;
    this->fnt = ( simu_time + SMALL ) / dt;
    this->nt = fnt;
}
