#pragma once
#include <vector>
#include <set>
#include <map>

class InterfaceSolver;
class BoundarySolver;
void GenerateGrid( int ni, float xmin, float xmax, float * xcoor );
void CfdSimulation( int ni, float * xcoor );
void CfdSolve( int zoneId, int ni, int nt, float cfl, float * xcoor, BoundarySolver * bcSolver );

const int BCInterface = -1;
const int BCInflow = 1;
const int BCOutflow = 2;

class IFaceBasic
{
public:
    IFaceBasic();
    ~IFaceBasic();
public:
    int face_id;
    int lc, rc;
    int count;
    int bctype;
};

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
    void FillBCPoints( std::set<int>& local_ptset );
    void MarkInterface();
    void MarkPhysicalBoundary();
    void SetBcType( int pt, int bcType );
public:
    int zoneId, nZones;
    int ni;
    std::map<int,IFaceBasic *> global_face_map;
    std::vector<int> bctypes;
    std::vector<int> bcpts;
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
