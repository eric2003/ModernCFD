#include "Cmpi.h"
#include "Simu.h"
#include "GpuSimu.h"

int main(int argc, char **argv)
{
    Cmpi::Init( argc, argv );

    Simu * simu = new Simu{};
    simu->Run();

    GpuSimu * gpu_simu = new GpuSimu{};
    gpu_simu->Run();
   
    delete simu;
    delete gpu_simu;

    Cmpi::Finalize();
    
    return 0;
}