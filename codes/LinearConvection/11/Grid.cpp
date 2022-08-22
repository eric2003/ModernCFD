#include "Grid.h"
#include <string>
#include <set>
#include <map>
#include "matplotlibcppModified.h"

namespace plt = matplotlibcpp;


BcTypeMap::BcTypeMap()
{
    ;
}

BcTypeMap::~BcTypeMap()
{
    ;
}

void BcTypeMap::Init()
{
    typedef std::pair< int, std::string > IntStringPair;

    this->bc_map.insert( IntStringPair( BCInterface, "BCInterface" ) );
    this->bc_map.insert( IntStringPair( BCInflow, "BCInflow" ) );
    this->bc_map.insert( IntStringPair( BCOutflow, "BCOutflow" ) );
}

std::string BcTypeMap::GetBcName( int bcId )
{
    return this->bc_map[ bcId ];
}

IFaceBasic::IFaceBasic()
{
    this->face_id = -1;
    this->lc = -1;
    this->rc = -1;
    this->count = 0;
}

IFaceBasic::~IFaceBasic()
{
}

void FreeMemory( std::map<int, IFaceBasic*>& global_ptmap )
{
    for ( auto it = global_ptmap.begin(); it != global_ptmap.end(); ++ it )
    {
        delete it->second;
    }
    global_ptmap.clear();
}

BoundarySolver::BoundarySolver()
{
    this->interfaceSolver = new InterfaceSolver();
}

BoundarySolver::~BoundarySolver()
{
    FreeMemory( this->global_face_map );
    delete this->interfaceSolver;
}

void Insert( std::set<int> &global_ptset, int i0 )
{
    std::set<int>::iterator it = global_ptset.find( i0 );
    if ( it == global_ptset.end() )
    {
        global_ptset.insert( i0 );
    }
}

void Insert( std::map<int, int> &global_ptmap, int i0 )
{
    std::map<int, int>::iterator it = global_ptmap.find( i0 );
    if ( it == global_ptmap.end() )
    {
        global_ptmap.insert( std::pair<int,int>(i0, 1) );
    }
    else
    {
        it->second ++;
    }
}

void Insert( std::map<int, int> &local_face_map, int local_pt, int global_pt )
{
    auto it = local_face_map.find( local_pt );
    if ( it == local_face_map.end() )
    {
        local_face_map.insert( std::pair<int,int>(local_pt, global_pt) );
    }
}

void Insert( std::map<int,IFaceBasic *> &global_ptmap, int iface_id, int zone_id )
{
    auto it = global_ptmap.find( iface_id );
    if ( it == global_ptmap.end() )
    {
        IFaceBasic* f = new IFaceBasic();
        f->face_id = iface_id;
        f->lc = zone_id;
        f->rc = -1;
        f->count ++;
        global_ptmap.insert( std::pair<int,IFaceBasic *>(iface_id, f) );
    }
    else
    {
        it->second->rc = zone_id;
        it->second->count ++;
    }
}

IFaceBasic * BoundarySolver::FindIFaceBasic( int id )
{
    auto it = global_face_map.find( id );
    if ( it != global_face_map.end() )
    {
        return it->second;
    }
    return 0;
}

void BoundarySolver::InsertVector(std::vector<int>& a, std::vector<int>& b )
{
    a.insert( a.end(), b.begin(), b.end() );
}

void BoundarySolver::FillBCPoints()
{
    for ( auto it = this->local_face_map.begin(); it != this->local_face_map.end(); ++ it )
    {
        int local_id = it->first;
        int global_id = it->second;
        IFaceBasic * f = FindIFaceBasic( global_id );
        if ( f->bctype == BCInterface )
        {
            this->interface_bcids.push_back( local_id );
            this->interface_bctypes.push_back( f->bctype );
        }
        else
        {
            this->physical_bcids.push_back( local_id );
            this->physical_bctypes.push_back( f->bctype );
        }
    }
    InsertVector( this->bctypes, this->physical_bctypes );
    InsertVector( this->bctypes, this->interface_bctypes );

    InsertVector( this->bcpts, this->physical_bcids );
    InsertVector( this->bcpts, this->interface_bcids );

    BcTypeMap bcTypeMap;
    bcTypeMap.Init();

    std::printf( "bcinfo......\n" );
    std::printf("local_face_map.size()=%zd\n", local_face_map.size() );
    std::printf("local_face_map.size()=%zd\n", local_face_map.size() );
    std::printf("physical_bctypes.size()=%zd\n", physical_bctypes.size() );
    std::printf("physical_bcids.size()=%zd\n", physical_bcids.size() );
    std::printf("interface_bctypes.size()=%zd\n", interface_bctypes.size() );
    std::printf("interface_bcids.size()=%zd\n", interface_bcids.size() );
    for ( int i = 0; i < this->bcpts.size(); ++ i )
    {
        int id = this->bcpts[ i ];
        int bctype = this->bctypes[ i ];
        std::string bcName = bcTypeMap.GetBcName( bctype );
        std::printf("point id=%d bctype=%d, bcName=%s\n", id, bctype, bcName.c_str() );
    }
    std::printf("\n");

}

void BoundarySolver::MarkInterface()
{
    for ( auto it = global_face_map.begin(); it != global_face_map.end(); ++ it )
    {
        if ( it->second->count == 2 )
        {
            it->second->bctype = BCInterface;
        }
    }
}

void BoundarySolver::MarkPhysicalBoundary()
{
    for ( auto it = global_face_map.begin(); it != global_face_map.end(); ++ it )
    {
        if ( it->second->count == 2 )
        {
            it->second->bctype = BCInterface;
        }
    }
}

void BoundarySolver::SetBcType( int pt, int bcType )
{
    auto it = global_face_map.find( pt );
    if ( it != global_face_map.end() )
    {
        it->second->bctype = bcType;
    }
}

void BoundarySolver::Init( int zoneId, int nZones, int ni )
{
    this->zoneId = zoneId;
    this->nZones = nZones;
    this->ni = ni;
    //local pt set : 30 40
    //global pt map : 0->1 10->2 20->2 30->2 40->1
    
    for ( int iZone = 0; iZone < nZones; ++ iZone )
    {
        int ishift = iZone * ( ni - 1 );
        int i0 = ishift + 1 + 0;
        int i1 = ishift + 1 + ni - 1;
        Insert( global_face_map, i0, iZone );
        Insert( global_face_map, i1, iZone );

        if ( iZone == zoneId )
        {
            int lpt = 1 + 0;
            int rpt = 1 + ni - 1;
            Insert( local_face_map, lpt, i0 );
            Insert( local_face_map, rpt, i1 );
        }
    }
    this->MarkInterface();

    int left_bcpt = 1;
    int right_bcpt = 1 + nZones * ( ni - 1 );
    this->SetBcType( left_bcpt, BCInflow );
    this->SetBcType( right_bcpt, BCOutflow );

    this->FillBCPoints();

    std::printf( "BoundarySolver::Init pt: zoneId = %d nZones = %d\n", zoneId, nZones );
    std::printf("global face map : \n" );
    for ( auto it = global_face_map.begin(); it != global_face_map.end(); ++ it )
    {
        std::printf( "%d->%d , lc = %d<-> rc = %d bcType = %d\n", it->first, it->second->count, it->second->lc, it->second->rc, it->second->bctype );
    }
    std::printf("\n");
    std::printf("local face map : \n" );
    for ( auto it = local_face_map.begin(); it != local_face_map.end(); ++ it )
    {
        std::printf( "%d -> %d \n", it->first, it->second );
    }
    std::printf("\n");
}

	
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
    int ist = 0;
    int ied = ni + 1;
    for ( int i = 1; i <= ni; ++ i )
    {
        float xm = xmin + ( i - 1 ) * dx;

        xcoor[ i ] = xm;
    }
    xcoor[ ist ] = 2 * xcoor[ ist + 1 ] - xcoor[ ist + 2 ];
    xcoor[ ied ] = 2 * xcoor[ ied - 1 ] - xcoor[ ied - 2 ];
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
    int ni_ghost = 2;
    int ni_total = ni + ni_ghost;

    for ( int i = 0; i < ni_total; ++ i )
    {
        float fm = SquareFun( xcoor[i] );
        q[i] = fm;
    }
}

void Visual( float * q, float * xcoor, int ni, const std::string & fileName )
{
    std::vector<float> qv{ q + 1, q + ni };
    std::vector<float> xv{ xcoor + 1, xcoor + ni };
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
    int ni_ghost = 2;
    int ni_total = ni + ni_ghost;
    float * q = new float[ ni_total ];
    InitField( q, xcoor, ni );
    Visual( q, xcoor, ni, "./cfd.png" );
    delete [] q;
}

void BoundaryInterface( BoundarySolver * bcSolver, float* q )
{
    int nIFace = bcSolver->GetNIFace();
    for (int i = 0; i < nIFace; ++ i )
    {
        int id = bcSolver->interface_bcids[ i ];
        q[ id ] = bcSolver->interface_bcvalues[ i ];
    }
}

void Boundary( float * q, float* xcoor, BoundarySolver * bcSolver )
{
    //physical boundary
    int nBFace = bcSolver->GetNBFace();
    std::printf(" Boundary zoneID = %d nBFace = %d\n", bcSolver->zoneId, nBFace);
    //std::printf(" bcSolver->bctypes.size() = %zd \n", bcSolver->bctypes.size());
    //std::printf(" bcSolver->bcpts.size() = %zd \n", bcSolver->bcpts.size());
    for ( int i = 0; i < nBFace; ++ i )
    {
        int bctype = bcSolver->bctypes[i];
        int id = bcSolver->bcpts[i];
        std::printf( " id = %d, bctype = %d \n", id, bctype );
        if ( bctype == BCInterface ) continue;
        if ( bctype == BCInflow )
        {
            float xm = xcoor[ id ];
            q[ id ] = SquareFun( xm );
        }
        else if ( bctype == BCOutflow )
        {
            ;
        }
    }

    //BoundaryInterface( bcSolver, q );
}

void SolveField( float * q, float * qn, int ni, int nt, float cfl, float * xcoor, BoundarySolver * bcSolver )
{
    int ni_ghost = 2;
    int ni_total = ni + ni_ghost;

    for ( int n = 0; n < nt; ++ n )
    {
        std::printf(" iStep = %d, nStep = %d \n", n + 1, nt);
        Boundary( q, xcoor, bcSolver );
        for ( int i = 0; i < ni_total; ++ i )
        {
            qn[i] = q[i];
        }

        for ( int i = 1; i < ni + 1; ++ i )
        {
            q[i] = qn[i] - cfl * ( qn[i] - qn[i-1] );
        }
    }
}

void CfdSolve( int zoneId, int ni, int nt, float cfl, float * xcoor, BoundarySolver * bcSolver )
{
    int ni_ghost = 2;
    int ni_total = ni + ni_ghost;

    float * q = new float[ ni_total ];
    float * qn = new float[ ni_total ];
    InitField( q, xcoor, ni );
    SolveField( q, qn, ni, nt, cfl, xcoor, bcSolver );
    char buffer[ 50 ];
    std::sprintf(buffer, "./cfd%d.png", zoneId );
    Visual( q, xcoor, ni, buffer );
    delete [] q;
    delete [] qn;
}
