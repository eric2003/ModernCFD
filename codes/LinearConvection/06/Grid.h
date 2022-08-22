#pragma once
#include <vector>

class Interface;
void GenerateGrid( int ni, float xmin, float xmax, float * xcoor );
void CfdSimulation( int ni, float * xcoor );
void CfdSolve(int zoneId, int ni, int nt, float cfl, float* xcoor, Interface* myinterface);

const int BCInterface = -1;
const int BCInflow = 1;
const int BCOutflow = 2;

class GlobalInterface
{
public:
    GlobalInterface();
    ~GlobalInterface();
};

class Interface
{
public:
    Interface();
    ~Interface();
public:
    int GetInterfaceCell( int iface );
    float GetInterfaceValue( int iface);
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
