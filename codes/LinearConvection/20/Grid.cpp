#include "Grid.h"
#include <string>
#include <set>
#include <map>
#include "matplotlibcppModified.h"
#include <mpi.h>

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

Interface_t::Interface_t()
{
    this->zone = -1;
    this->cell = -1;
    this->ghost_cell = -1;
    this->face = -1;
}

Interface_t::~Interface_t()
{
    ;
}

IFaceBasic::IFaceBasic()
{
    this->global_face_id = -1;
    this->count = 0;
    this->bctype = -1;
}

IFaceBasic::~IFaceBasic()
{
}

int IFaceBasic::GetNeighborZone()  const
{
    return this->face_pair[ this->nid ].zone;
}

int IFaceBasic::GetNeighborFace()  const
{
    return this->face_pair[ this->nid ].face;
}

void FreeMemory( std::map<int, IFaceBasic*>& global_ptmap )
{
    for ( auto it = global_ptmap.begin(); it != global_ptmap.end(); ++ it )
    {
        delete it->second;
    }
    global_ptmap.clear();
}


PhysicalBoundarySolver::PhysicalBoundarySolver()
{
}

PhysicalBoundarySolver::~PhysicalBoundarySolver()
{
}

void PhysicalBoundarySolver::Init()
{
    for ( int i = 0; i < this->physical_bc_list.size(); ++ i )
    {
        IFaceBasic *f = this->physical_bc_list[ i ];
        Interface_t * fp = & f->face_pair[ 0 ];
        this->physical_faceids.push_back( fp->face );
        this->physical_ghost_cells.push_back( fp->ghost_cell );
        this->physical_bctypes.push_back( f->bctype );
    }
}

BoundarySolver::BoundarySolver()
{
    this->interfaceSolver = new InterfaceSolver();
    this->physicalSolver = new PhysicalBoundarySolver();
}

BoundarySolver::~BoundarySolver()
{
    FreeMemory( this->global_face_map );
    delete this->interfaceSolver;
    delete this->physicalSolver;
}

int BoundarySolver::GetNIFace()
{ 
    return interfaceSolver->GetNIFace();
}

void Insert( std::map<int, int> &local_face_map, int local_pt, int global_pt )
{
    auto it = local_face_map.find( local_pt );
    if ( it == local_face_map.end() )
    {
        local_face_map.insert( std::pair<int,int>(local_pt, global_pt) );
    }
}

void Insert( std::map<int, IFaceBasic *> & global_face_map, int global_face_id, int local_face_id, int zone_id, int cell_id, int ghost_cell_id )
{
    auto it = global_face_map.find( global_face_id );
    if ( it == global_face_map.end() )
    {
        IFaceBasic* f = new IFaceBasic();
        f->global_face_id = global_face_id;
        int idx = 0;
        f->face_pair[ idx ].zone = zone_id;
        f->face_pair[ idx ].cell = cell_id;
        f->face_pair[ idx ].ghost_cell = ghost_cell_id;
        f->face_pair[ idx ].face = local_face_id;
        f->count ++;
        global_face_map.insert( std::pair<int,IFaceBasic *>(global_face_id, f) );
    }
    else
    {
        IFaceBasic * f = it->second;
        int idx = 1;
        f->face_pair[ idx ].zone = zone_id;
        f->face_pair[ idx ].cell = cell_id;
        f->face_pair[ idx ].ghost_cell = ghost_cell_id;
        f->face_pair[ idx ].face = local_face_id;
        f->count ++;
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
    this->interfaceSolver->zone_id = this->zoneId;
    for ( auto it = this->local_face_map.begin(); it != this->local_face_map.end(); ++ it )
    {
        int local_face_id = it->first;
        int global_face_id = it->second;
        IFaceBasic * f = FindIFaceBasic( global_face_id );

        if ( f->bctype == BCInterface )
        {
            this->interfaceSolver->interlist.push_back( f );
        }
        else
        {
            this->physicalSolver->physical_bc_list.push_back( f );
        }
    }
    this->interfaceSolver->Init();
    this->physicalSolver->Init();

    BcTypeMap bcTypeMap;
    bcTypeMap.Init();

    InsertVector( this->bctypes, this->physicalSolver->physical_bctypes );
    InsertVector( this->bctypes, this->interfaceSolver->interface_bctypes );
    InsertVector( this->bc_faceids, this->physicalSolver->physical_faceids );
    InsertVector( this->bc_faceids, this->interfaceSolver->interface_faceids );
    InsertVector( this->bc_ghostcells, this->physicalSolver->physical_ghost_cells );
    InsertVector( this->bc_ghostcells, this->interfaceSolver->interface_ghost_cells );



    std::printf( "bcinfo......\n" );
    std::printf("local_face_map.size()=%zd\n", local_face_map.size() );
    std::printf("local_face_map.size()=%zd\n", local_face_map.size() );
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
        IFaceBasic *f = it->second;
        Interface_t * f1 = &f->face_pair[ 0 ];
        Interface_t * f2 = &f->face_pair[ 1 ];
        std::printf( "%d->%d ", it->first, f->count );
        std::printf( "(zone,cell,ghost)[(%d,%d,%d),(%d,%d,%d)] bcType = %d\n", \
            f1->zone, f1->cell, f1->ghost_cell, \
            f2->zone, f2->cell, f2->ghost_cell, \
            f->bctype );
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
        Insert( global_face_map, global_face_id0, local_face_id0, iZone, cell_id0, ghost_cell_id0 );
        Insert( global_face_map, global_face_id1, local_face_id1, iZone, cell_id1, ghost_cell_id1 );

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

void IData::Print()
{
    std::printf( " IData::Print() zone_id =%d \n", zone_id );
    for ( int i = 0; i < this->interface_ids.size(); ++ i )
    {
        std::printf( "%d ", this->interface_ids[ i ] );
    }
    std::printf( "\n" );
}

InterfaceSolver::InterfaceSolver()
{
    ;
}

InterfaceSolver::~InterfaceSolver()
{
    ;
}

//bool CmpIFaceBasicPtr(const IFaceBasic * a, const IFaceBasic * b)
//{
//    if ( a->neighbor_zone != b->neighbor_zone )
//    {
//        return a->neighbor_zone < b->neighbor_zone;
//    }
//    return a->neighbor_face < b->neighbor_face;
//}

bool CmpIFaceBasicPtr(const IFaceBasic * a, const IFaceBasic * b)
{
    int a_neighbor_zone = a->GetNeighborZone();
    int b_neighbor_zone = b->GetNeighborZone();
    if ( a_neighbor_zone != b_neighbor_zone )
    {
        return a_neighbor_zone < b_neighbor_zone;
    }
    return a->GetNeighborFace() < b->GetNeighborFace();
}

void InterfaceSolver::SortInterface()
{
    std::printf( " SetNeighborData nSize =%zd \n", this->interlist.size() );
    for ( int i = 0; i < this->interlist.size(); ++ i )
    {
        IFaceBasic *f = this->interlist[ i ];
        if ( f->face_pair[0].zone == this->zone_id )
        {
            f->nid = 1;
        }
        else
        {
            f->nid = 0;
        }
    }
    std::sort( this->interlist.begin(), this->interlist.end(), CmpIFaceBasicPtr );
    for ( int i = 0; i < this->interlist.size(); ++ i )
    {
        IFaceBasic *f = this->interlist[ i ];
        std::printf( " i =%d neighbor_zone = %d neighbor_face = %d \n", i, f->face_pair[f->nid].zone, f->face_pair[f->nid].face);
    }
}

void InterfaceSolver::SetNeighborData()
{
    std::printf( " InterfaceSolver::SetNeighborData this->interlist.size() =%zd \n", this->interlist.size() );
    int nSize = this->interlist.size();
    std::printf( " SetNeighborData nSize =%d \n", nSize );
    if ( nSize == 0 ) return;

    int ist = 0;
    int icount = 0;
    while ( true )
    {
        std::vector<int> interfaces;
        int neighbor_zone_id_start = this->interlist[ ist ]->GetNeighborZone();
        int ipos = this->neighbor_datas.size();
        for ( int i = ist; i < this->interlist.size(); ++ i )
        {
            IFaceBasic * f = this->interlist[ i ];
            int neighbor_zone_now = f->GetNeighborZone();
            if ( neighbor_zone_id_start != neighbor_zone_now )
            {
                ist = i;
                break;
            }
            int jpos = interfaces.size();
            int neighbor_face = f->GetNeighborFace();
            this->interface_neighbor_zoneids.push_back( neighbor_zone_id_start );
            this->interface_neighbor_faceids.push_back( neighbor_face );
            this->interface_ipos.push_back( ipos );
            this->interface_jpos.push_back( jpos );
            interfaces.push_back( neighbor_face );
            icount ++;
        }
        IData iData;
        iData.zone_id = neighbor_zone_id_start;
        iData.interface_ids = interfaces;
        this->neighbor_datas.push_back( iData );
        if ( icount == this->interlist.size() ) break;
    }
}

float InterfaceSolver::GetBcValue( int iface )
{
    int ipos = this->interface_ipos[ iface ];
    int jpos = this->interface_jpos[ iface ];
    return this->neighbor_datas[ ipos ].interface_values[ jpos ];
}

void InterfaceSolver::Init()
{
    std::printf( " InterfaceSolver::Init()\n" );
    this->SortInterface();
    this->SetNeighborData();

    std::printf( " myzoneId =%d neighbor_datas.size() = %zd --------------------\n", zone_id, neighbor_datas.size() );
    for ( int i = 0; i < this->neighbor_datas.size(); ++ i )
    {
        this->neighbor_datas[ i ].Print();
    }
    this->SwapData();
}

void InterfaceSolver::SwapData()
{
    for ( int i = 0; i < this->neighbor_datas.size(); ++ i )
    {
        IData & idata = this->neighbor_datas[ i ];
        int neighbor_zoneid = idata.zone_id;
        int ip = neighbor_zoneid;
        int tag = 0;
        std::printf( " 1£º this->zone_id =%d neighbor_zoneid = %d\n", this->zone_id, neighbor_zoneid );
        if ( neighbor_zoneid > this->zone_id )
        {
            int n = idata.interface_ids.size();
            MPI_Send(&n, 1, MPI_INT, ip, tag, MPI_COMM_WORLD);
            for ( int i = 0; i < idata.interface_ids.size(); ++ i )
            {
                std::printf( "1: idata.interface_ids[%d]=%d \n", i, idata.interface_ids[ i ] );
            }
            MPI_Send(idata.interface_ids.data(), n, MPI_INT, ip, tag, MPI_COMM_WORLD);
        }
        else
        {
            int n = -1;
            std::printf( " 1 £º n =%d \n", n );
            MPI_Recv(&n, 1, MPI_INT, ip, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            std::printf( " 1 recv £º n =%d \n", n );
            std::vector<int> tmp_ids(n,-1);
            for ( int i = 0; i < tmp_ids.size(); ++ i )
            {
                std::printf( "1 000: tmp_ids[%d]=%d \n", i, tmp_ids[ i ] );
            }
            MPI_Recv(tmp_ids.data(), n, MPI_INT, ip, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for ( int i = 0; i < tmp_ids.size(); ++ i )
            {
                std::printf( "1 111: tmp_ids[%d]=%d \n", i, tmp_ids[ i ] );
            }
        }
    }

    for ( int i = 0; i < this->neighbor_datas.size(); ++ i )
    {
        IData & idata = this->neighbor_datas[ i ];
        int neighbor_zoneid = idata.zone_id;
        int ip = neighbor_zoneid;
        int tag = 0;
        std::printf( " 2£º this->zone_id =%d neighbor_zoneid = %d\n", this->zone_id, neighbor_zoneid );
        if ( neighbor_zoneid > this->zone_id )
        {
            int n = -1;
            std::printf( " 2 £º n =%d \n", n );
            MPI_Recv(&n, 1, MPI_INT, ip, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            std::printf( " 2 recv £º n =%d \n", n );
            std::vector<int> tmp_ids(n,-1);
            for ( int i = 0; i < tmp_ids.size(); ++ i )
            {
                std::printf( "2 000: tmp_ids[%d]=%d \n", i, tmp_ids[ i ] );
            }
            MPI_Recv(tmp_ids.data(), n, MPI_INT, ip, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for ( int i = 0; i < tmp_ids.size(); ++ i )
            {
                std::printf( "2 111: tmp_ids[%d]=%d \n", i, tmp_ids[ i ] );
            }
        }
        else
        {
            int n = idata.interface_ids.size();
            MPI_Send(&n, 1, MPI_INT, ip, tag, MPI_COMM_WORLD);
            for ( int i = 0; i < idata.interface_ids.size(); ++ i )
            {
                std::printf( "2: idata.interface_ids[%d]=%d \n", i, idata.interface_ids[ i ] );
            }
            MPI_Send(idata.interface_ids.data(), n, MPI_INT, ip, tag, MPI_COMM_WORLD);
        }
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
    for ( int iface = 0; iface < nIFace; ++ iface )
    {
        int ghostcell_id = interfaceSolver->interface_ghost_cells[ iface ];
        int ipos = interfaceSolver->interface_ipos[ iface ];
        int jpos = interfaceSolver->interface_jpos[ iface ];
        int neighbor_zone_id = interfaceSolver->interface_neighbor_zoneids[ iface ];
        int neighbor_face_id = interfaceSolver->interface_neighbor_faceids[ iface ];
        std::printf( " ipos = %d, jpos = %d, zoneid = %d, neighbor_zone_id = %d, neighbor_face_id = %d \n",\
            ipos, jpos, interfaceSolver->zone_id, neighbor_zone_id, neighbor_face_id );
        //float bcvalue = interfaceSolver->neighbor_datas[ipos].interface_values[jpos];
        //float bcvalue = interfaceSolver->GetBcValue( iface );
        //q[ ghostcell_id ] = bcvalue;
    }
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
