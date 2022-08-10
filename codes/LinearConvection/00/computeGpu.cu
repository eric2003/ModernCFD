#include <iostream>
#include "computeGpu.h"

// Device code
__global__ void sqrtKernel( float *a, float *b )
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    b[ tid ] = sqrt( a[ tid ] );
}

// Initialize an array with random data (between 0 and 1)
void initData( float *data, int nElems )
{
    for ( int i = 0; i < nElems; ++ i )
    {
        data[ i ] = static_cast<float>( rand() ) / RAND_MAX;
    }
}

void computeGPU( float *host_d, int blockSize, int gridSize )
{
    int dataSize = blockSize * gridSize;
    size_t nSize = dataSize * sizeof(float);
    
    float *d_a = NULL;
    cudaMalloc( (void **)&d_a, nSize );
    
    float *d_b = NULL;
    cudaMalloc( (void **)&d_b, nSize );
    
    cudaMemcpy( d_a, host_d, nSize, cudaMemcpyHostToDevice );
    
    sqrtKernel<<<gridSize, blockSize>>>( d_a, d_b );
    
    cudaMemcpy( host_d, d_b, nSize, cudaMemcpyDeviceToHost );
    
    cudaFree( d_a );
    cudaFree( d_b );
}

float sum( float *data, int size )
{
    float accum = 0.0f;
    
    for ( int i = 0; i < size; ++ i )
    {
        accum += data[i];
    }
    
    return accum;
}
