#include "GpuSimu.h"
#include "Cmpi.h"
#include "computeGpu.h"
#include <iostream>

GpuSimu::GpuSimu()
{
    ;
}

GpuSimu::~GpuSimu()
{
    ;
}

void GpuSimu::Init(int argc, char **argv)
{
}

void GpuSimu::Run()
{
    this->blockSize = 256;
    this->gridSize = 10000;
    this->dataSizePerNode = gridSize * blockSize;
    this->dataSizeTotal = dataSizePerNode * Cmpi::nproc;
    this->dataRoot = NULL;
    if ( Cmpi::pid == 0 )
    {
        std::cout << "Running on " << Cmpi::nproc << " nodes" << std::endl;
        dataRoot = new float[ dataSizeTotal ];
        initData( dataRoot, dataSizeTotal );
    }
    // Allocate a buffer on each node
    this->dataNode = new float[ this->dataSizePerNode ];

    // Dispatch a portion of the input data to each node
    MPI_Scatter(dataRoot, this->dataSizePerNode, MPI_FLOAT, this->dataNode, this->dataSizePerNode, MPI_FLOAT, 0, MPI_COMM_WORLD);

    if ( Cmpi::pid == 0 )
    {
        // No need for root data any more
        delete [] this->dataRoot;
    }

    // On each node, run computation on GPU
    computeGPU( this->dataNode, this->blockSize, this->gridSize );

    // Reduction to the root node, computing the sum of output elements
    float sumNode = sum( dataNode, dataSizePerNode );
    float sumRoot;
    std::cout << "sumNode = " << sumNode << " process id = " << Cmpi::pid << " Cmpi::nproc = " << Cmpi::nproc << std::endl;

    MPI_Reduce(&sumNode, &sumRoot, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

    if ( Cmpi::pid == 0 )
    {
        float average = sumRoot / this->dataSizeTotal;
        std::cout << "Average of square roots is: " << average << std::endl;
        std::cout << "PASSED\n";
    }

    delete [] dataNode;
}