#include <cstdio>

__global__ void kernelPrint()
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( idx == 0 )
    {
        std::printf("blockDim.x.y.z=[%d,%d,%d]\n", blockDim.x, blockDim.y, blockDim.z);
		std::printf("gridDim.x.y.z=[%d,%d,%d]\n", gridDim.x, gridDim.y, gridDim.z);
    }
}

int main(int argc, char* argv[])
{
    kernelPrint<<<1, 10 >>>();
	kernelPrint<<<5, 10 >>>();

    cudaDeviceSynchronize();

    return 0;
}