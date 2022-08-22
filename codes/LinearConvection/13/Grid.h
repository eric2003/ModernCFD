#pragma once
#include <vector>
#include <set>
#include <map>
#include <string>

class InterfaceSolver;
class BoundarySolver;
void GenerateGrid( int ni, float xmin, float xmax, float * xcoor );
void CfdSimulation( int ni, float * xcoor );
void CfdSolve( int zoneId, int ni, int nt, float cfl, float * xcoor, BoundarySolver * bcSolver );

const int BCInterface = -1;
const int BCInflow = 1;
const int BCOutflow = 2;

class BcTypeMap
{
public:
    BcTypeMap();
    ~BcTypeMap();
public:
    void Init();
    std::string GetBcName( int bcId );
private:
    std::map< int, std::string > bc_map;
};

class IFaceBasic
{
public:
    IFaceBasic();
    ~IFaceBasic();
public:
    int face_id;
    int cell1, cell2;
    int ghost_cell1, ghost_cell2;
    int zone1, zone2;
    int count;
    int bctype;
};

struct IdPair
{
public:
    int idl, idg; //local->global
};

class BoundarySolver
{
public:
    BoundarySolver();
    ~BoundarySolver();
public:
    int GetNBFace() { return bctypes.size(); };
    int GetNIFace() { return interface_bctypes.size(); };
public:
    void Init( int zoneId, int nZones, int ni );
    void FillBCPoints();
    void MarkInterface();
    void PrintBcInfo();
    void MarkPhysicalBoundary();
    void SetBcType( int bc_face_id, int bcType );
    IFaceBasic * FindIFaceBasic( int id );
    void InsertVector( std::vector<int> & a, std::vector<int> & b );
public:
    int zoneId, nZones;
    int ni;
    std::map<int,IFaceBasic *> global_face_map;
    std::map<int,int> local_face_map;
    InterfaceSolver* interfaceSolver;
public:
    std::vector<int> bctypes;
    std::vector<int> bc_faceids;
    std::vector<int> bc_ghostcells;
    std::vector<int> physical_faceids;
    std::vector<int> physical_ghost_cells;
    std::vector<int> physical_bctypes;
    std::vector<int> interface_faceids;
    std::vector<int> interface_ghost_cells;
    std::vector<int> interface_bctypes;
    std::vector<float> interface_bcvalues;
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
