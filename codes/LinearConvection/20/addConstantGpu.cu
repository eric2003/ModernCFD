#include <omp.h>
#include <cstdio>
#include "addConstantGpu.h"

__global__ void kernelAddConstant(int *g_a, const int b)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    g_a[ idx ] += b;
}

int CheckResult( int *data, const int n, const int b )
{
    for ( int i = 0; i < n; ++ i )
    {
        if ( data[i] != i + b ) return 0;
    }
    
    return 1;
}

void addConstantGpu()
{
    std::printf("Starting addConstantGpu...\n\n");

    int num_gpus = 0;    
    cudaGetDeviceCount( &num_gpus );
    
    if ( num_gpus < 1 ) {
        std::printf("no CUDA capable devices were detected\n");
        exit(1);
    }
    
    std::printf("number of host CPUs:\t%d\n", omp_get_num_procs());
    std::printf("number of CUDA devices:\t%d\n", num_gpus);
    
    for ( int i = 0; i < num_gpus; ++ i )
    {
        cudaDeviceProp dprop;
        cudaGetDeviceProperties( &dprop, i);
        std::printf("   %d: %s\n", i, dprop.name);
    }

    std::printf("---------------------------\n");

    unsigned int n = num_gpus * 8192;
    unsigned int nbytes = n * sizeof(int);

    int * a = static_cast<int *>( std::malloc( nbytes ) );
    int b   = 3;    
    
    for ( unsigned int i = 0; i < n; ++ i )
    {
        a[i] = i;
    }

    int nCpuThreads = 2 * num_gpus;
    int N_per_cpu_thread = n / nCpuThreads;
    unsigned int nbytes_per_cpu_thread = nbytes / nCpuThreads;
    
    omp_set_num_threads( nCpuThreads );
#pragma omp parallel
    {
        unsigned int cpu_thread_id = omp_get_thread_num();
        unsigned int num_cpu_threads = omp_get_num_threads();
        int gpu_id = -1;
        cudaSetDevice( cpu_thread_id % num_gpus );
        cudaGetDevice( &gpu_id );
        std::printf("CPU thread %d (of %d) uses CUDA device %d\n", cpu_thread_id,num_cpu_threads, gpu_id);
        int *d_a = 0;
        int *sub_a = a + cpu_thread_id * N_per_cpu_thread;
        
        int block_size = 128;
        int grid_size = ( N_per_cpu_thread + block_size - 1 ) / block_size;
        dim3 block_dim( block_size );  // 128 threads per block
        dim3 grid_dim( grid_size );
        std::printf("block_size=%d \n", block_size);
        std::printf("grid_size=%d \n", grid_size);
        
        cudaMalloc( (void **)&d_a, nbytes_per_cpu_thread );
        cudaMemset( d_a, 0, nbytes_per_cpu_thread );
        cudaMemcpy( d_a, sub_a, nbytes_per_cpu_thread, cudaMemcpyHostToDevice );
        kernelAddConstant<<<grid_dim, block_dim>>>( d_a, b );
        
        cudaMemcpy( sub_a, d_a, nbytes_per_cpu_thread, cudaMemcpyDeviceToHost );
        cudaFree( d_a );
    }
    bool bResult = CheckResult(a, n, b);
    if ( bResult )
    {
        std::printf( "CHECK PASSED!\n" );
    }
    else
    {
        std::printf( "CHECK FAILED!\n" );
    }
    std::free(a);

}
