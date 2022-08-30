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
    void AllocateField( Geom * geom );
    void DeallocateField( Geom * geom );
    void SolveField( CfdPara * cfd_para, Geom * geom );
    void SaveField( CfdPara * cfd_para, Geom * geom );
    void InitField( CfdPara * cfd_para,  Geom * geom );
    void Visualize( CfdPara * cfd_para, Geom * geom );
    void Boundary( float * q, Geom * geom );
    void BoundaryInterface( float * q, Geom * geom );
    void Timestep( CfdPara * cfd_para, Geom * geom );
private:
    void SetInflowField( CfdPara * cfd_para, Geom * geom );
    void ReadField( CfdPara * cfd_para, Geom * geom );
public:
    float * q;
    float * qn;
    float * timestep;

};

