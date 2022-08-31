#include <cstdio>
#include "print.h"
#ifdef ENABLE_CUDA
#include "print_cuda.h"
#endif


void PrintCpu();

void PrintGeneral()
{
#ifdef ENABLE_CUDA
    PrintGuda();
#else
    PrintCpu();
#endif
}

void PrintCpu()
{
    std::printf("void PrintCpu()\n");
}