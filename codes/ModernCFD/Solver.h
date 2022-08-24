#pragma once

class Geom;
class CfdPara;

class Solver
{
public:
    Solver();
    ~Solver();
public:
    void Init();
    void Run( CfdPara * cfd_para, Geom * geom );
    void CfdSolve( CfdPara * cfd_para, Geom * geom );
public:

};

