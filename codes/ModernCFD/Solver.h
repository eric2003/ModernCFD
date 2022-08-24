#pragma once

class Geom;
class CfdPara;
class BoundarySolver;

class Solver
{
public:
    Solver();
    ~Solver();
public:
    void Init();
    void Run( CfdPara * cfd_para, Geom * geom, BoundarySolver * bcSolver, int zoneId );
public:

};

