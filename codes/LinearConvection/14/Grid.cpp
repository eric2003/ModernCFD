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
    this->cell1 = -1;
    this->cell2 = -1;
    this->ghost_cell1 = -1;
    this->ghost_cell2 = -1;
    this->zone1 = -1;
    this->zone2 = -1;
    this->count = 0;
    this->bctype = -1;
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

int BoundarySolver::GetNIFace()
{ 
    return interfaceSolver->interface_bctypes.size();
}

void Insert( std::map<int, int> &local_face_map, int local_pt, int global_pt )
{
    auto it = local_face_map.find( local_pt );
    if ( it == local_face_map.end() )
    {
        local_face_map.insert( std::pair<int,int>(local_pt, global_pt) );
    }
}

void Insert( std::map<int, IFaceBasic *> & global_ptmap, int iface_id, int zone_id, int cell_id, int ghost_cell_id )
{
    auto it = global_ptmap.find( iface_id );
    if ( it == global_ptmap.end() )
    {
        IFaceBasic* f = new IFaceBasic();
        f->face_id = iface_id;
        f->zone1 = zone_id;
        f->cell1 = cell_id;
        f->ghost_cell1 = ghost_cell_id;
        f->zone2 = -1;
        f->cell2 = -1;
        f->ghost_cell2 = -1;
        f->count ++;
        global_ptmap.insert( std::pair<int,IFaceBasic *>(iface_id, f) );
    }
    else
    {
        it->second->zone2 = zone_id;
        it->second->cell2 = cell_id;
        it->second->ghost_cell2 = ghost_cell_id;
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
        int local_face_id = it->first;
        int global_face_id = it->second;
        IFaceBasic * f = FindIFaceBasic( global_face_id );
        int neighbor_zone = -1;
        int ghost_cell = -1;
        if ( f->zone1 == zoneId )
        {
            ghost_cell = f->ghost_cell1;
            neighbor_zone = f->zone2;
        }
        else
        {
            ghost_cell = f->ghost_cell2;
            neighbor_zone = f->zone1;
        }

        if ( f->bctype == BCInterface )
        {
            this->interfaceSolver->neighbor_zones.insert( neighbor_zone );
            this->interfaceSolver->interlist.push_back( f );
            this->interfaceSolver->interface_faceids.push_back( local_face_id );
            this->interfaceSolver->interface_ghost_cells.push_back( ghost_cell );
            this->interfaceSolver->interface_bctypes.push_back( f->bctype );
        }
        else
        {
            this->physical_faceids.push_back( local_face_id );
            this->physical_ghost_cells.push_back( ghost_cell );
            this->physical_bctypes.push_back( f->bctype );
        }
    }
    InsertVector( this->bctypes, this->physical_bctypes );
    InsertVector( this->bctypes, this->interfaceSolver->interface_bctypes );
    InsertVector( this->bc_faceids, this->physical_faceids );
    InsertVector( this->bc_faceids, this->interfaceSolver->interface_faceids );
    InsertVector( this->bc_ghostcells, this->physical_ghost_cells );
    InsertVector( this->bc_ghostcells, this->interfaceSolver->interface_ghost_cells );

    this->interfaceSolver->Init();

    BcTypeMap bcTypeMap;
    bcTypeMap.Init();

    std::printf( "bcinfo......\n" );
    std::printf("local_face_map.size()=%zd\n", local_face_map.size() );
    std::printf("local_face_map.size()=%zd\n", local_face_map.size() );
    std::printf("physical_bctypes.size()=%zd\n", physical_bctypes.size() );
    std::printf("physical_faceids.size()=%zd\n", physical_faceids.size() );
    std::printf("interfaceSolver->interface_bctypes.size()=%zd\n", interfaceSolver->interface_bctypes.size() );
    std::printf("interfaceSolver->interface_faceids.size()=%zd\n", interfaceSolver->interface_faceids.size() );
    for ( int iface = 0; iface < this->bc_faceids.size(); ++ iface )
    {
        int bc_faceid = this->bc_faceids[ iface ];
        int bctype = this->bctypes[ iface ];
        std::string bcName = bcTypeMap.GetBcName( bctype );
        std::printf("bc_faceid=%d bctype=%d, bcName=%s\n", bc_faceid, bctype, bcName.c_str() );
    }
    std::printf("\n");

}

void BoundarySolver::PrintBcInfo()
{
    std::printf( "BoundarySolver::Init pt: zoneId = %d nZones = %d\n", zoneId, nZones );
    std::printf("global face map : \n" );
    for ( auto it = global_face_map.begin(); it != global_face_map.end(); ++ it )
    {
        std::printf( "%d->%d ", it->first, it->second->count );
        std::printf( "(zone,cell,ghost)[(%d,%d,%d),(%d,%d,%d)] bcType = %d\n", \
            it->second->zone1, it->second->cell1, it->second->ghost_cell1, \
            it->second->zone2, it->second->cell2, it->second->ghost_cell2, \
            it->second->bctype );
    }
    std::printf("\n");
    std::printf("local face map : \n" );
    for ( auto it = local_face_map.begin(); it != local_face_map.end(); ++ it )
    {
        std::printf( "%d -> %d \n", it->first, it->second );
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

void BoundarySolver::SetBcType( int bc_face_id, int bcType )
{
    auto it = global_face_map.find( bc_face_id );
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
        int local_face_id0 = 1 + 0;
        int local_face_id1 = 1 + ni - 1;
        int global_face_id0 = ishift + 1 + 0;
        int global_face_id1 = ishift + 1 + ni - 1;
        //int cell_id0 = global_face_id0 + 1;
        //int cell_id1 = global_face_id1 - 1;
        //int ghost_cell_id0 = global_face_id0 - 1;
        //int ghost_cell_id1 = global_face_id1 + 1;
        int cell_id0 = local_face_id0 + 1;
        int cell_id1 = local_face_id1 - 1;
        int ghost_cell_id0 = local_face_id0 - 1;
        int ghost_cell_id1 = local_face_id1 + 1;
        Insert( global_face_map, global_face_id0, iZone, cell_id0, ghost_cell_id0 );
        Insert( global_face_map, global_face_id1, iZone, cell_id1, ghost_cell_id1 );

        if ( iZone == zoneId )
        {
            Insert( local_face_map, local_face_id0, global_face_id0 );
            Insert( local_face_map, local_face_id1, global_face_id1 );
        }
    }
    this->MarkInterface();

    int left_bcface_id = 1;
    int right_bcface_id = 1 + nZones * ( ni - 1 );
    this->SetBcType( left_bcface_id, BCInflow );
    this->SetBcType( right_bcface_id, BCOutflow );

    this->FillBCPoints();

    this->PrintBcInfo();
}

	
GlobalInterface::GlobalInterface()
{
    ;
}

GlobalInterface::~GlobalInterface()
{
    ;
}

IData::IData()
{
    ;
}

IData::~IData()
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


void InterfaceSolver::Init()
{
    std::printf( " InterfaceSolver::Init()\n" );
    //std::printf( " neighbor_zones.size() = %zd\n", neighbor_zones.size() );
    //for ( auto it = this->neighbor_zones.begin(); it != this->neighbor_zones.end(); ++ it )
    //{
    //    std::printf( "%d ", *it );
    //}
    //std::printf( "\n" );

    int nIFace = this->interface_bctypes.size();
    std::printf( " nIFace = %d\n", nIFace );
    for ( int iface = 0; iface < nIFace; ++ iface )
    {
        int ghostcell_id = this->interface_ghost_cells[ iface ];
        int bc_faceid = this->interface_faceids[ iface ];
        int bctype = this->interface_bctypes[ iface ];
        //float bcvalue = this->interface_bcvalues[ iface ];
        //std::printf( " ghostcell_id = %d, bc_faceid = %d, bctype = %d, bcvalue = %f, x = %f\n", ghostcell_id, bc_faceid, bctype, bcvalue );
        //q[ ghostcell_id ] = this->interface_bcvalues[ iface ];
    }
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

void BoundaryInterface( float * q, float* xcoor, BoundarySolver * bcSolver )
{
    int nIFace = bcSolver->GetNIFace();
    std::printf( " BoundaryInterface\n" );
    std::printf( " nIFace = %d\n", nIFace );
    InterfaceSolver * interfaceSolver = bcSolver->interfaceSolver;
    //for ( int iface = 0; iface < nIFace; ++ iface )
    //{
    //    int ghostcell_id = interfaceSolver->interface_ghost_cells[ iface ];
    //    int bc_faceid = interfaceSolver->interface_faceids[ iface ];
    //    int bctype = interfaceSolver->interface_bctypes[ iface ];
    //    float bcvalue = interfaceSolver->interface_bcvalues[ iface ];
    //    std::printf( " ghostcell_id = %d, bc_faceid = %d, bctype = %d, bcvalue = %f, x = %f\n", ghostcell_id, bc_faceid, bctype, bcvalue, xcoor[ ghostcell_id ] );
    //    q[ ghostcell_id ] = interfaceSolver->interface_bcvalues[ iface ];
    //}
}

void Boundary( float * q, float* xcoor, BoundarySolver * bcSolver )
{
    //physical boundary
    int nBFace = bcSolver->GetNBFace();
    std::printf(" Boundary zoneID = %d nBFace = %d\n", bcSolver->zoneId, nBFace);
    for ( int i = 0; i < nBFace; ++ i )
    {
        int bctype = bcSolver->bctypes[ i ];
        int ghostcell_id = bcSolver->bc_ghostcells[ i ];
        int bc_faceid = bcSolver->bc_faceids[ i ];
        std::printf( " ghostcell_id = %d, bc_faceid = %d, bctype = %d x = %f\n", ghostcell_id, bc_faceid, bctype, xcoor[ ghostcell_id ] );
        if ( bctype == BCInterface ) continue;
        if ( bctype == BCInflow )
        {
            float xm = xcoor[ ghostcell_id ];
            q[ ghostcell_id ] = SquareFun( xm );
        }
        else if ( bctype == BCOutflow )
        {
            q[ ghostcell_id ] = q[ bc_faceid ];
        }
    }

    BoundaryInterface( q, xcoor, bcSolver );
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
