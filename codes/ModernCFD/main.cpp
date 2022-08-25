#include "Cmpi.h"
#include "Simu.h"

int main(int argc, char **argv)
{
    Cmpi::Init( argc, argv );

    Simu * simu = new Simu{};

    simu->Run();
   
    delete simu;
    Cmpi::Finalize();
    
    return 0;
}