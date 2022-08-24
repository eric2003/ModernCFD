#include <omp.h>
#include <cstdio>
#include <mpi.h>
#include <iostream>
#include <set>
#include "computeGpu.h"
#include "addConstantGpu.h"
//#include "Grid.h"
#include "Cmpi.h"
#include "Simu.h"

int main(int argc, char **argv)
{
    int blockSize = 256;
    int gridSize = 10000;
    int dataSizePerNode = gridSize * blockSize;

    Cmpi::Init( argc, argv );

    int dataSizeTotal = dataSizePerNode * Cmpi::nproc;
    float *dataRoot = NULL;

    Simu * simu = new Simu{};
    simu->Run();
    ////cfd parameter
    //CfdPara * cfd_para = new CfdPara{};
    //cfd_para->Init();

    //int nZones = Cmpi::nproc;
    //Geom * geom = new Geom();
    //geom->Init( nZones );
    //int zoneId = Cmpi::pid;
   
    //BoundarySolver bcSolver;
    //bcSolver.zoneId = zoneId;
    //bcSolver.Init( zoneId, nZones, geom->ni );
    //geom->GenerateGrid( cfd_para );
    //CfdSolve( geom->ni_global, geom->xcoor_global, cfd_para->simu_time, zoneId, geom->ni, cfd_para->nt, cfd_para->cfl, geom->xcoor, &bcSolver );

    if ( Cmpi::pid == 0 )
    {
        std::cout << "Running on " << Cmpi::nproc << " nodes" << std::endl;
        dataRoot = new float[ dataSizeTotal ];
        initData( dataRoot, dataSizeTotal );
    }

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
    
    if ( Cmpi::pid == 0 )
    {
        float average = sumRoot / dataSizeTotal;
        std::cout << "Average of square roots is: " << average << std::endl;
        std::cout << "PASSED\n";
    }
    
    delete [] dataNode;
    //delete cfd_para;
    //delete geom;
    delete simu;

    Cmpi::Finalize();

    
    return 0;
}