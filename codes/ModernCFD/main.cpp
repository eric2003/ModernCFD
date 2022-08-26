#include "Simu.h"

int main(int argc, char **argv)
{
    Simu * simu = new Simu{ argc, argv };

    simu->Run();
   
    delete simu;
    
    return 0;
}