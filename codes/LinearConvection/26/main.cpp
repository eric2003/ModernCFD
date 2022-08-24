#include <omp.h>
#include <cstdio>
#include <mpi.h>
#include <iostream>
#include <set>
#include "computeGpu.h"
#include "addConstantGpu.h"
#include "Grid.h"
#include "Cmpi.h"
#define SMALL 1.0e-10

class CfdPara
{
public:
    CfdPara();
    ~CfdPara();
public:
    void Init();
public:
    int nt;
    float cfl;
    float simu_time;
    float xlen;
    float dx;
    float cspeed;
    float dt;
    float fnt;
    float xmin, xmax;

};

CfdPara::CfdPara()
{
    //this->Init();
}

CfdPara::~CfdPara()
{
    ;
}

void CfdPara::Init()
{
    int ni_global = 41;
    this->xmin = 0.0;
    this->xmax = 2.0;

    this->cfl = 0.5;
    this->simu_time = 0.625;
    this->xlen = xmax - xmin;
    this->dx = xlen / ( ni_global - 1 );
    this->cspeed = 1.0;
    this->dt = dx * cfl / cspeed;
    this->fnt = ( simu_time + SMALL ) / dt;
    this->nt = fnt;
}

int main(int argc, char **argv)
{
    int blockSize = 256;
    int gridSize = 10000;
    int dataSizePerNode = gridSize * blockSize;

    Cmpi::Init( argc, argv );

    int dataSizeTotal = dataSizePerNode * Cmpi::nproc;
    float *dataRoot = NULL;
    //cfd parameter
    CfdPara * cfd_para = new CfdPara{};
    cfd_para->Init();
    int nZones = Cmpi::nproc;
    int ni_ghost = 2;
    int ni_global = 41;
    int ni_global_total = ni_global + ni_ghost;
    //float xmin = 0.0;
    //float xmax = 2.0;
    float * xcoor_global = 0;
    int zoneId = Cmpi::pid;
    int ni = ( ni_global - 1 ) / nZones + 1;
    int ist = 1;
    int ied = ni;
    int ni_total = ni + ni_ghost;
    float * xcoor = new float[ ni_total ];
    
    //float cfl = 0.5;
    //float simu_time = 0.625;
    //float xlen = xmax - xmin;
    //float dx = xlen / ( ni_global - 1 );
    //float cspeed = 1.0;
    //float dt = dx * cfl / cspeed;
    //float fnt = ( simu_time + SMALL ) / dt;
    //int nt = fnt;

    BoundarySolver bcSolver;
    bcSolver.zoneId = zoneId;
    bcSolver.Init( zoneId, nZones, ni );

    if ( Cmpi::pid == 0 )
    {
        std::printf( " dt = %f, dx = %f, nt = %d, ni_global = %d\n", cfd_para->dt, cfd_para->dx, cfd_para->nt, ni_global );
    }
    
    if ( Cmpi::pid == 0 )
    {
        std::cout << "Running on " << Cmpi::nproc << " nodes" << std::endl;
        dataRoot = new float[ dataSizeTotal ];
        initData( dataRoot, dataSizeTotal );
        xcoor_global = new float[ ni_global_total ];
        GenerateGrid( ni_global, cfd_para->xmin, cfd_para->xmax, xcoor_global );
		
        for ( int i = 0; i < ni_total; ++ i )
        {
            xcoor[ i ] = xcoor_global[ i ];
        }
        for ( int ip = 1; ip < Cmpi::nproc; ++ ip )
        {
            int istart = ip * ( ni - 1 );
            float * source_start = xcoor_global + istart;
            MPI_Send(source_start, ni_total, MPI_FLOAT, ip, 0, MPI_COMM_WORLD );
        }
    }
    else
    {
        MPI_Recv(xcoor, ni_total, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    
     std::printf("print xcoor: process id = %d Cmpi::nproc = %d\n", Cmpi::pid, Cmpi::nproc );
     for ( int i = 0; i < ni_total; ++ i )
     {
         std::printf("%f ", xcoor[ i ] );
     }
     std::printf("\n");
    
    // Allocate a buffer on each node
    float *dataNode = new float[ dataSizePerNode ];
    
    // Dispatch a portion of the input data to each node
    MPI_Scatter(dataRoot, dataSizePerNode, MPI_FLOAT, dataNode, dataSizePerNode, MPI_FLOAT, 0, MPI_COMM_WORLD);
    
    if ( Cmpi::pid == 0 )
    {
        // No need for root data any more
        delete[] dataRoot;
    }
    
    // On each node, run computation on GPU
    computeGPU( dataNode, blockSize, gridSize );
    addConstantGpu();
    
    // Reduction to the root node, computing the sum of output elements
    float sumNode = sum( dataNode, dataSizePerNode );
    float sumRoot;
    std::cout << "sumNode = " << sumNode << " process id = " << Cmpi::pid << " Cmpi::nproc = " << Cmpi::nproc << std::endl;
    
    MPI_Reduce(&sumNode, &sumRoot, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
    
    CfdSolve( ni_global, xcoor_global, cfd_para->simu_time, zoneId, ni, cfd_para->nt, cfd_para->cfl, xcoor, &bcSolver );

    if ( Cmpi::pid == 0 )
    {
        float average = sumRoot / dataSizeTotal;
        std::cout << "Average of square roots is: " << average << std::endl;
        std::cout << "PASSED\n";
        delete[] xcoor_global;
    }
    
    delete [] xcoor;
    delete [] dataNode;
    delete cfd_para;

    Cmpi::Finalize();

    
    return 0;
}