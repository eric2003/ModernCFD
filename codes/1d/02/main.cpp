#include <omp.h>
#include <iostream>

typedef float Real;
#define SMALL 1.0e-10
class FieldPara
{
public:
    FieldPara(){};
    ~FieldPara(){};
public:
    int nx;
    int nt;

    float len;
    float dx;
    float dt;
    float c;
    float cfl;
	float time;
public:
    void PrintInfo()
    {
		this->nx = 41;
        this->len = 2.0;
		this->dx = this->len / ( this->nx - 1.0 ); 
		this->nt = 25;
		this->dt = 0.025;
		this->c  = 1.0;
		this->cfl = this->c * this->dt / this->dx;
		std::cout << " this->c = " << this->c << std::endl;
		std::cout << " this->dt = " << this->dt << std::endl;
		std::cout << " this->dx = " << this->dx << std::endl;
		std::cout << " this->len = " << this->len << std::endl;
		std::cout << " this->cfl = " << this->cfl << std::endl;
    };
    void Init()
    {
		this->cfl = 0.5;
		this->time = 0.625;
        this->len = 2.0;
		this->nx = 41;
		this->dx = this->len / ( this->nx - 1.0 );
		this->c  = 1.0;
		this->dt = this->dx * cfl / this->c;
		float fnt =( this->time + SMALL ) / this->dt;
		this->nt = fnt;
		std::cout << " this->c = " << this->c << std::endl;
		std::cout << " this->dt = " << this->dt << std::endl;
		std::cout << " this->dx = " << this->dx << std::endl;
		std::cout << " this->len = " << this->len << std::endl;
		std::cout << " this->cfl = " << this->cfl << std::endl;
		std::cout << " this->time = " << this->time << std::endl;
		std::cout << " this->nt = " << this->nt << std::endl;
    };	
};

int main(int argc, char** argv)
{
    FieldPara fp;
	fp.Init();
    return 0;
}
