#pragma once
#include <vector>

class BoundarySolver;

class Geom_t
{
public:
    Geom_t();
    ~Geom_t();
public:
    static void Init();
public:
    static int ni_ghost;
    static int ni_global;
    static int ni_global_total;
public:
    static std::vector<int> zone_nis;
    static std::vector<int> proc_ids;
    static std::vector<int> zone_ids;
};

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
