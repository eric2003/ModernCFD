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
    void SolveField( CfdPara * cfd_para, Geom * geom );
    void InitField( Geom * geom );
    void Visualize( CfdPara * cfd_para, Geom * geom );
    void Boundary( float * q, Geom * geom );
    void BoundaryInterface( float * q, Geom * geom );
public:
    float * q;
    float * qn;

};

