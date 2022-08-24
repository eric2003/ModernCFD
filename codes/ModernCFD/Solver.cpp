#include "Solver.h"
#include <string>
#include <set>
#include <map>
#include <mpi.h>
#include "Cmpi.h"
#include "Grid.h"
#include "Geom.h"
#include "CfdPara.h"
#include "matplotlibcppModified.h"
namespace plt = matplotlibcpp;

float SquareFun( float xm )
{
    if ( xm >= 0.5 && xm <= 1.0 )
    {
        return 2.0;
    }
    return 1.0;
}

void Theory( float time, float c, std::vector<float>& theory, std::vector<float>& xcoor )
{
    int ni = xcoor.size();
    float xs = c * time;
    for ( int i = 0; i < ni; ++ i )
    {
        float xm = xcoor[i];
        float xm_new = xm - xs;
        float fm = SquareFun( xm_new );
        theory[i] = fm;
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

void Visual( std::vector<float> & q, std::vector<float> & theory, std::vector<float> & x,  const std::string & fileName )
{
    // Set the size of output image to 1200x780 pixels
    plt::figure_size(1200, 780);
    // Plot line from given x and y data. Color is selected automatically.
    plt::plot(x, q, {{"label", "calc"}});
    plt::plot(x, theory, {{"label", "theory"}});
    // Add graph title
    plt::title("1d convection");
    plt::xlabel("x");
    plt::ylabel("u");
    // Enable legend.
    plt::legend();

    // Save the image (file format is determined by the extension)
    plt::savefig( fileName.c_str() );
}

Solver::Solver()
{
}

Solver::~Solver()
{
    ;
}

void Solver::Init()
{
}

void Solver::Run( CfdPara * cfd_para, Geom * geom )
{
    this->CfdSolve( cfd_para, geom );
}

void Solver::InitField( Geom * geom )
{
    for ( int i = 0; i < geom->ni_total; ++ i )
    {
        float fm = SquareFun( geom->xcoor[ i ] );
        this->q[ i ] = fm;
    }
}

void Solver::CfdSolve( CfdPara * cfd_para, Geom * geom )
{
    this->q = new float[ geom->ni_total ];
    this->qn = new float[ geom->ni_total ];
    this->InitField( geom );
    this->SolveField( cfd_para, geom );
    this->Visualize( cfd_para, geom );
    delete [] this->q;
    delete [] this->qn;
}

void Solver::SolveField( CfdPara * cfd_para, Geom * geom )
{
    for ( int n = 0; n < cfd_para->nt; ++ n )
    {
        if ( geom->zoneId == 0 )
        {
            std::printf(" iStep = %d, nStep = %d \n", n + 1, cfd_para->nt);
        }

        this->Boundary( q, geom );
        for ( int i = 0; i < geom->ni_total; ++ i )
        {
            qn[ i ] = q[ i ];
        }

        for ( int i = 1; i < geom->ni + 1; ++ i )
        {
            q[ i ] = qn[ i ] - cfd_para->cfl * ( qn[ i ] - qn[ i - 1 ] );
        }
    }
}

void Solver::Boundary( float * q, Geom * geom )
{
    BoundarySolver * bcSolver = geom->bcSolver;
    //physical boundary
    int nBFace = bcSolver->GetNBFace();
    //std::printf(" Boundary zoneID = %d nBFace = %d\n", bcSolver->zoneId, nBFace);
    for ( int iface = 0; iface < nBFace; ++ iface )
    {
        int bctype = bcSolver->bctypes[ iface ];
        int ghostcell_id = bcSolver->bc_ghostcells[ iface ];
        int bc_faceid = bcSolver->bc_faceids[ iface ];
        if ( bctype == BCInterface ) continue;
        if ( bctype == BCInflow )
        {
            float xm = geom->xcoor[ ghostcell_id ];
            q[ ghostcell_id ] = SquareFun( xm );
        }
        else if ( bctype == BCOutflow )
        {
            q[ ghostcell_id ] = q[ bc_faceid ];
        }
    }

    this->BoundaryInterface( q, geom );
}

void Solver::BoundaryInterface( float * q, Geom * geom )
{
    BoundarySolver * bcSolver = geom->bcSolver;
    int nIFace = bcSolver->GetNIFace();
    //std::printf( " BoundaryInterface nIFace = %d\n", nIFace );
    InterfaceSolver * interfaceSolver = bcSolver->interfaceSolver;
    interfaceSolver->SwapData( q );
    for ( int iface = 0; iface < nIFace; ++ iface )
    {
        int ghostcell_id = interfaceSolver->interface_ghost_cells[ iface ];
        //interfaceSolver->ShowInfo( iface );
        float bcvalue = interfaceSolver->GetBcValue( iface );
        q[ ghostcell_id ] = bcvalue;
    }
}

void Solver::Visualize( CfdPara * cfd_para, Geom * geom )
{
    char buffer[ 50 ];
    std::sprintf( buffer, "./cfd%d.png", geom->zoneId );
    Visual( this->q, geom->xcoor, geom->ni, buffer );

    std::vector<float> q_global;
    std::vector<float> x_global;
    int root = 0;
    int tag = 0;
    if ( geom->zoneId != 0 )
    {
        MPI_Send( this->q, geom->ni_total, MPI_FLOAT, root, tag, MPI_COMM_WORLD );
    }
    else
    {
        std::vector<std::vector<float>> qvec( Cmpi::nproc );
        for ( int ip = 1; ip < Cmpi::nproc; ++ ip )
        {
            qvec[ ip ].resize( geom->ni_total );
        }
        qvec[ 0 ].insert( qvec[ 0 ].end(), this->q, this->q + geom->ni_total );

        for ( int ip = 1; ip < Cmpi::nproc; ++ ip )
        {
            MPI_Recv( qvec[ ip ].data(), geom->ni_total, MPI_FLOAT, ip, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE );
        }

        for ( int ip = 0; ip < Cmpi::nproc; ++ ip )
        {
            if ( ip == 0 )
            {
                q_global.insert( q_global.end(), qvec[ ip ].begin() + 1, qvec[ ip ].end() - 1 );
            }
            else
            {
                q_global.insert( q_global.end(), qvec[ ip ].begin() + 2, qvec[ ip ].end() - 1 );
            }
        }
        x_global.insert( x_global.end(), geom->xcoor_global + 1, geom->xcoor_global + geom->ni_global + 1 );
        std::vector<float> theory;
        theory.resize( x_global.size() );
        Theory( cfd_para->simu_time, cfd_para->cspeed, theory, x_global );
        Visual( q_global, theory, x_global, "./cfd.png" );
    }
}

