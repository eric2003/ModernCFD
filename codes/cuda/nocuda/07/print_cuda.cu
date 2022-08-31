#include <cstdio>

#ifdef ENABLE_CUDA
__global__ void kernelPrint()
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( idx == 0 )
    {
        std::printf("blockDim.x.y.z=[%d,%d,%d]\n", blockDim.x, blockDim.y, blockDim.z);
        std::printf("gridDim.x.y.z=[%d,%d,%d]\n", gridDim.x, gridDim.y, gridDim.z);
    }
}

void PrintGuda()
{
    std::printf("start void PrintGuda()------------------------\n");

    kernelPrint<<<1, 10 >>>();
    kernelPrint<<<5, 10 >>>();

    cudaDeviceSynchronize();
    std::printf("end void PrintGuda()------------------------\n");
}

#endif