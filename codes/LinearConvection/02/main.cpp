#include <omp.h>
#include <cstdio>
#include <mpi.h>
#include <iostream>
#include "computeGpu.h"
#include "addConstantGpu.h"
#include "Grid.h"

int main(int argc, char *argv[])
{
    int blockSize = 256;
    int gridSize = 10000;
    int dataSizePerNode = gridSize * blockSize;
    
    MPI_Init(&argc,&argv); 
    int myid, nproc;
    MPI_Comm_rank(MPI_COMM_WORLD,&myid); 
    MPI_Comm_size(MPI_COMM_WORLD,&nproc); 

    int dataSizeTotal = dataSizePerNode * nproc;
    float *dataRoot = NULL;
    //cfd parameter
    int ni = 41;
    float xmin = 0.0;
    float xmax = 2.0;
    float * xcoor_global = 0;
    
    if ( myid == 0 )
    {
        std::cout << "Running on " << nproc << " nodes" << std::endl;
        dataRoot = new float[ dataSizeTotal ];
        initData( dataRoot, dataSizeTotal );
        xcoor_global = new float[ ni ];
        GenerateGrid( ni, xmin, xmax, xcoor_global );
    }
    
    // Allocate a buffer on each node
    float *dataNode = new float[ dataSizePerNode ];
    
    // Dispatch a portion of the input data to each node
    MPI_Scatter(dataRoot, dataSizePerNode, MPI_FLOAT, dataNode, dataSizePerNode, MPI_FLOAT, 0, MPI_COMM_WORLD);
    
    if ( myid == 0 )
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
    std::cout << "sumNode = " << sumNode << " process id = " << myid << " nproc = " << nproc << std::endl;
    
    MPI_Reduce(&sumNode, &sumRoot, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
    
    if ( myid == 0 )
    {
        float average = sumRoot / dataSizeTotal;
        std::cout << "Average of square roots is: " << average << std::endl;
        std::cout << "PASSED\n";
        CfdSimulation( ni, xcoor_global );
        delete[] xcoor_global;
    }
    
    delete[] dataNode;
    MPI_Finalize();
    
    return 0;
}