#pragma once
#include <vector>

class InterfaceSolver;
class BoundarySolver;
void GenerateGrid( int ni, float xmin, float xmax, float * xcoor );
void CfdSimulation( int ni, float * xcoor );
void CfdSolve( int zoneId, int ni, int nt, float cfl, float * xcoor, BoundarySolver * bcSolver );

const int BCInterface = -1;
const int BCInflow = 1;
const int BCOutflow = 2;

class BoundarySolver
{
public:
    BoundarySolver();
    ~BoundarySolver();
public:
    int GetNBFace() { return bctypes.size(); };
    int GetNIFace();// { return interfaceSolver->GetNIFace(); };
public:
    void Init( int zoneId, int nZones, int ni );
public:
    int zoneId, nZones;
    int ni;
    std::vector<int> bctypes;
    InterfaceSolver* interfaceSolver;
};

class GlobalInterface
{
public:
    GlobalInterface();
    ~GlobalInterface();
};

class InterfaceSolver
{
public:
    InterfaceSolver();
    ~InterfaceSolver();
public:
    int GetInterfaceCell( int iface );
    float GetInterfaceValue( int iface );
    int GetNIFace() { return ifaces.size(); }
public:
    int zoneId;
    int bctype;
    std::vector< int > iglobalfaces;
    //local interface id
    std::vector< int > ifaces;
    std::vector< int > cells;
public:
    std::vector< float > q;
};
