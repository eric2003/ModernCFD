#pragma once
class CfdPara;

class BoundarySolver;
class Geom
{
public:
    Geom();
    ~Geom();
public:
    void Init();
    void GenerateGrid();
    void GenerateGrid( int ni, float xmin, float xmax, float * xcoor );
    void ComputeGeom();
public:
    int zoneId;
    int nZones;
    int ni_ghost;
    int ni_global;
    int ni_global_total;
    int ni;
    int ni_total;
    float * xcoor_global;
    float * xcoor;
public:
    float xlen;
    float dx;
    float * ds;
    float xmin, xmax;
public:
    BoundarySolver * bcSolver;
};
