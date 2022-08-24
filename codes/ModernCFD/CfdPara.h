#pragma once
#include <vector>
#include <set>
#include <map>
#include <string>

#define SMALL 1.0e-10

class CfdPara
{
public:
    CfdPara();
    ~CfdPara();
public:
    void Init();
public:
    int nt;
    float cfl;
    float simu_time;
    float xlen;
    float dx;
    float cspeed;
    float dt;
    float fnt;
    float xmin, xmax;

};
