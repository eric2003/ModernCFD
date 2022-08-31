#pragma once
#include <vector>
#include <set>
#include <map>
#include <string>

#define SMALL 1.0e-10

class Geom;

class CfdPara
{
public:
    CfdPara();
    ~CfdPara();
public:
    void Init( Geom * geom );
public:
    int nt;
    int irestart; //0 restart, 1 continue
    float cfl;
    float simu_time;
    float cspeed;
    float dt;
    float fnt;
};
