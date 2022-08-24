#include "Geom.h"
#include "Cmpi.h"
#include "Grid.h"
#include <iostream>

Geom::Geom()
{
    this->xcoor_global = 0;
    this->xcoor = 0;
}

Geom::~Geom()
{
    delete [] this->xcoor_global;
    delete [] this->xcoor;
}

void Geom::Init()
{
    this->nZones = Cmpi::nproc;
    this->ni_ghost = 2;
    this->ni_global = 41;
    this->ni_global_total = this->ni_global + this->ni_ghost;
    this->ni = ( this->ni_global - 1 ) / this->nZones + 1;
    this->ni_total = this->ni + this->ni_ghost;
    this->xcoor_global = 0;
    this->xcoor = new float[ this->ni_total ];

    this->zoneId = Cmpi::pid;
    this->bcSolver = new BoundarySolver{};
    this->bcSolver->zoneId = zoneId;
    this->bcSolver->Init( zoneId, nZones, this->ni );

    this->xmin = 0.0;
    this->xmax = 2.0;

    this->xlen = xmax - xmin;
    this->dx = xlen / ( ni_global - 1 );
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
        //std::printf( " dt = %f, dx = %f, nt = %d, ni_global = %d\n", cfd_para->dt, cfd_para->dx, cfd_para->nt, this->ni_global );
    }

    if ( Cmpi::pid == 0 )
    {
        std::cout << "Running on " << Cmpi::nproc << " nodes" << std::endl;
        this->xcoor_global = new float[ this->ni_global_total ];
        this->GenerateGrid( this->ni_global, this->xmin, this->xmax, this->xcoor_global );

        for ( int i = 0; i < this->ni_total; ++ i )
        {
            this->xcoor[ i ] = this->xcoor_global[ i ];
        }
        for ( int ip = 1; ip < Cmpi::nproc; ++ ip )
        {
            int istart = ip * ( this->ni - 1 );
            float * source_start = this->xcoor_global + istart;
            MPI_Send( source_start, this->ni_total, MPI_FLOAT, ip, 0, MPI_COMM_WORLD );
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