#include <omp.h>
#include <cstdio>
#include <mpi.h>
#include <iostream>
#include <set>
#include "computeGpu.h"
#include "addConstantGpu.h"
#include "Cmpi.h"
#include "Simu.h"
#include "GpuSimu.h"

int main(int argc, char **argv)
{
    //int blockSize = 256;
    //int gridSize = 10000;
    //int dataSizePerNode = gridSize * blockSize;

    Cmpi::Init( argc, argv );

    //int dataSizeTotal = dataSizePerNode * Cmpi::nproc;
    //float *dataRoot = NULL;

    Simu * simu = new Simu{};
    simu->Run();

    GpuSimu * gpu_simu = new GpuSimu{};
    gpu_simu->Run();

    //if ( Cmpi::pid == 0 )
    //{
    //    std::cout << "Running on " << Cmpi::nproc << " nodes" << std::endl;
    //    dataRoot = new float[ dataSizeTotal ];
    //    initData( dataRoot, dataSizeTotal );
    //}

    //// Allocate a buffer on each node
    //float *dataNode = new float[ dataSizePerNode ];
    //
    //// Dispatch a portion of the input data to each node
    //MPI_Scatter(dataRoot, dataSizePerNode, MPI_FLOAT, dataNode, dataSizePerNode, MPI_FLOAT, 0, MPI_COMM_WORLD);
    //
    //if ( Cmpi::pid == 0 )
    //{
    //    // No need for root data any more
    //    delete[] dataRoot;
    //}
    
    //// On each node, run computation on GPU
    //computeGPU( dataNode, blockSize, gridSize );
    addConstantGpu();
    
    //// Reduction to the root node, computing the sum of output elements
    //float sumNode = sum( dataNode, dataSizePerNode );
    //float sumRoot;
    //std::cout << "sumNode = " << sumNode << " process id = " << Cmpi::pid << " Cmpi::nproc = " << Cmpi::nproc << std::endl;
    //
    //MPI_Reduce(&sumNode, &sumRoot, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
    //
    //if ( Cmpi::pid == 0 )
    //{
    //    float average = sumRoot / dataSizeTotal;
    //    std::cout << "Average of square roots is: " << average << std::endl;
    //    std::cout << "PASSED\n";
    //}
    //
    //delete [] dataNode;
    delete simu;
    delete gpu_simu;

    Cmpi::Finalize();

    
    return 0;
}