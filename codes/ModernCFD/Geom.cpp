#include "Geom.h"
#include "Cmpi.h"
#include "Grid.h"
#include <iostream>

int Geom_t::ni_ghost = 2;
int Geom_t::ni_global = 1;
int Geom_t::ni_global_total = -1;
std::vector<int> Geom_t::zonenis;

Geom_t::Geom_t()
{
}

Geom_t::~Geom_t()
{
}

void Geom_t::Init()
{
    //Geom_t::ni_global = 41;
    Geom_t::ni_global = 42;
    Geom_t::ni_ghost = 2;
    Geom_t::ni_global_total = Geom_t::ni_global + Geom_t::ni_ghost;

    int nZones = Cmpi::nproc;
    Geom_t::zonenis.resize( nZones );
    int grid_ni = ( Geom_t::ni_global + nZones - 1 ) / nZones;
    int ni_last = Geom_t::ni_global - ( nZones - 1 ) * ( grid_ni - 1 );

    for ( int i = 0; i < nZones - 1; ++ i )
    {
        Geom_t::zonenis[i] = grid_ni;
    }
    Geom_t::zonenis[nZones - 1] = ni_last;
    std::printf( "zone ni----------------------\n" );
    for ( int i = 0; i < nZones; ++ i )
    {
        std::printf( "%d ", Geom_t::zonenis[i] );
    }
    std::printf( "\n" );
}

Geom::Geom()
{
    this->xcoor_global = 0;
    this->xcoor = 0;
    this->ds = 0;
}

Geom::~Geom()
{
    delete [] this->xcoor_global;
    delete [] this->xcoor;
    delete [] this->ds;
}

void Geom::Init()
{
    this->nZones = Cmpi::nproc;
    this->zoneId = Cmpi::pid;
    this->ni = Geom_t::zonenis[ this->zoneId ];
    this->ni_total = this->ni + Geom_t::ni_ghost;
    this->xcoor_global = 0;
    this->xcoor = new float[ this->ni_total ];
    this->ds = new float[ this->ni_total ];

    this->bcSolver = new BoundarySolver{};
    this->bcSolver->zoneId = zoneId;
    this->bcSolver->Init( zoneId, nZones, this->ni );

    this->xmin = 0.0;
    this->xmax = 2.0;

    this->xlen = xmax - xmin;
    this->dx = xlen / ( Geom_t::ni_global - 1 );
}

void Geom::GenerateGrid( int ni, float xmin, float xmax, float * xcoor )
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

void Geom::GenerateGrid()
{
    if ( Cmpi::pid == 0 )
    {
        std::cout << "Running on " << Cmpi::nproc << " nodes" << std::endl;
        this->xcoor_global = new float[ Geom_t::ni_global_total ];
        this->GenerateGrid( Geom_t::ni_global, this->xmin, this->xmax, this->xcoor_global );

        for ( int i = 0; i < this->ni_total; ++ i )
        {
            this->xcoor[ i ] = this->xcoor_global[ i ];
        }
        int istart = 0;
        for ( int ip = 1; ip < Cmpi::nproc; ++ ip )
        {
            int ni_tmp = Geom_t::zonenis[ ip ];
            int ni_total_tmp = ni_tmp + Geom_t::ni_ghost;
            istart += ( ni_tmp - 1 );
            float * source_start = this->xcoor_global + istart;
            MPI_Send( source_start, ni_total_tmp, MPI_FLOAT, ip, 0, MPI_COMM_WORLD );
        }
    }
    else
    {
        MPI_Recv(this->xcoor, this->ni_total, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    std::printf("print xcoor: process id = %d Cmpi::nproc = %d\n", Cmpi::pid, Cmpi::nproc );
    for ( int i = 0; i < this->ni_total; ++ i )
    {
        std::printf("%f ", this->xcoor[ i ] );
    }
    std::printf("\n");
}

void Geom::ComputeGeom()
{
    for ( int i = 1; i < this->ni_total - 1; ++ i )
    {
        this->ds[ i ] = this->xcoor[ i ] - this->xcoor[ i - 1 ];
    }

    this->ds[ 0 ] = this->ds[ 1 ];
    this->ds[ this->ni + 1 ] = this->ds[ this->ni ];
}
