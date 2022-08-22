#include "Grid.h"
#include <string>
#include <set>
#include <map>
#include "matplotlibcppModified.h"

namespace plt = matplotlibcpp;

BoundarySolver::BoundarySolver()
{
    this->interfaceSolver = new InterfaceSolver();
}

BoundarySolver::~BoundarySolver()
{
    delete this->interfaceSolver;
}

void Insert( std::set<int> &glbal_ptset, int i0 )
{
    std::set<int>::iterator it = glbal_ptset.find( i0 );
    if ( it == glbal_ptset.end() )
    {
        glbal_ptset.insert( i0 );
    }
}

void Insert( std::map<int, int> &glbal_ptmap, int i0 )
{
    std::map<int, int>::iterator it = glbal_ptmap.find( i0 );
    if ( it == glbal_ptmap.end() )
    {
        glbal_ptmap.insert( std::pair<int,int>(i0, 1) );
    }
    else
    {
        it->second ++;
    }
}

void BoundarySolver::Init( int zoneId, int nZones, int ni )
{
    this->zoneId = zoneId;
    this->nZones = nZones;
    this->ni = ni;
    std::set<int> glbal_ptset;
    std::set<int> local_ptset;
    std::vector<int> ptarray;
    std::map<int,int> global_ptmap;
    std::map<int,int> local_ptmap;
    std::vector<int> bctypes;
    for ( int iZone = 0; iZone < nZones; ++ iZone )
    {
        int ishift = iZone * ( ni - 1 );
        int i0 = ishift + 0;
        int i1 = ishift + ni - 1;
        Insert( global_ptmap, i0 );
        Insert( global_ptmap, i1 );

        if ( iZone == zoneId )
        {
            Insert( local_ptset, i0 );
            Insert( local_ptset, i1 );
        }
    }

    std::printf("BoundarySolver::Init pt: zoneId = %d nZones = %d\n", zoneId, nZones );
    std::printf("local pt set : " );
    for ( auto it = local_ptset.begin(); it != local_ptset.end(); ++ it )
    {
        std::printf("%d ", *it );
    }
    std::printf("\n");
    std::printf("global pt map : " );
    for ( auto it = global_ptmap.begin(); it != global_ptmap.end(); ++ it )
    {
        std::printf( "%d->%d ", it->first, it->second );
    }
    std::printf("\n");
}

int BoundarySolver::GetNIFace()
{ 
	return interfaceSolver->GetNIFace();
};
	
GlobalInterface::GlobalInterface()
{
    ;
}

GlobalInterface::~GlobalInterface()
{
    ;
}

InterfaceSolver::InterfaceSolver()
{
    ;
}

InterfaceSolver::~InterfaceSolver()
{
    ;
}

int InterfaceSolver::GetInterfaceCell(int iface)
{
    //0 1
    //0 ni
    return cells[iface];
}

float InterfaceSolver::GetInterfaceValue(int iface)
{
    return q[iface];
}

void GenerateGrid( int ni, float xmin, float xmax, float * xcoor )
{
    float dx = ( xmax - xmin ) / ( ni - 1 );

    for ( int i = 0; i < ni; ++ i )
    {
        float xm = xmin + i * dx;

        xcoor[ i ] = xm;
    }
}

float SquareFun( float xm )
{
    if ( xm >= 0.5 && xm <= 1.0 )
    {
        return 2.0;
    }
    return 1.0;
}       

void InitField( float * q, float * xcoor, int ni )
{
    for ( int i = 0; i < ni; ++ i )
    {
        float fm = SquareFun( xcoor[i] );
        q[i] = fm;
    }
}

void Visual( float * q, float * xcoor, int ni, const std::string & fileName )
{
    std::vector<float> qv{ q, q + ni };
    std::vector<float> xv{ xcoor, xcoor + ni };
    // Set the size of output image to 1200x780 pixels
    plt::figure_size(1200, 780);
    // Plot line from given x and y data. Color is selected automatically.
    plt::plot(xv, qv, {{"label", "calc"}});
    // Add graph title
    plt::title("1d convection");
    plt::xlabel("x");
    plt::ylabel("u");
    // Enable legend.
    plt::legend();
    
    // Save the image (file format is determined by the extension)
    plt::savefig( fileName.c_str() );
}

void CfdSimulation( int ni, float * xcoor )
{
    float * q = new float[ ni ];
    InitField( q, xcoor, ni );
    Visual( q, xcoor, ni, "./cfd.png" );
    delete [] q;
}

void BoundaryInterface( BoundarySolver * bcSolver, float* q )
{
    InterfaceSolver* interfaceSolver = bcSolver->interfaceSolver;
    //Interface boundary
    int nIFace = interfaceSolver->GetNIFace();
    for (int iface = 0; iface < nIFace; ++iface)
    {
        int pt = interfaceSolver->GetInterfaceCell(iface);
        q[pt] = interfaceSolver->GetInterfaceValue(iface);
    }
}

void Boundary( float * q, float* xcoor, BoundarySolver * bcSolver )
{
    //physical boundary
    int nBFace = bcSolver->GetNBFace();
    std::printf(" Boundary zoneID = %d nBFace = %d\n", bcSolver->zoneId, nBFace);
    for ( int i = 0; i < nBFace; ++ i )
    {
        int bctype = bcSolver->bctypes[i];
        if ( bctype == BCInterface ) continue;
        if ( bctype == BCInflow )
        {
            float xm = xcoor[i];
            q[i] = SquareFun( xm );
        }
        else if ( bctype == BCOutflow )
        {
            ;
        }
    }

    BoundaryInterface( bcSolver, q );
}

void SolveField( float * q, float * qn, int ni, int nt, float cfl, float * xcoor, BoundarySolver * bcSolver )
{
    for ( int n = 0; n < nt; ++ n )
    {
        std::printf(" iStep = %d, nStep = %d \n", n + 1, nt);
        Boundary( q, xcoor, bcSolver );
        for ( int i = 0; i < ni; ++ i )
        {
            qn[i] = q[i];
        }

        for ( int i = 1; i < ni; ++ i )
        {
            q[i] = qn[i] - cfl * ( qn[i] - qn[i-1] );
        }
    }
}

void CfdSolve( int zoneId, int ni, int nt, float cfl, float * xcoor, BoundarySolver * bcSolver )
{
    float * q = new float[ ni ];
    float * qn = new float[ ni ];
    InitField( q, xcoor, ni );
    SolveField( q, qn, ni, nt, cfl, xcoor, bcSolver);
    char buffer[ 50 ];
    std::sprintf(buffer, "./cfd%d.png", zoneId );
    Visual( q, xcoor, ni, buffer );
    delete [] q;
    delete [] qn;
}
