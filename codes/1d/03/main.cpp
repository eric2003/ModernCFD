#include <omp.h>
#include <iostream>
#include <vector>

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

void GenerateGrid( int ni, float xmin, float xmax, std::vector<float> &xcoor )
{
	float dx = ( xmax - xmin ) / ( ni - 1 );

	for ( int i = 0; i < ni; ++ i )
	{
		float xm = xmin + i * dx;

		xcoor[ i ] = xm;
	}
}

float SquareFun( float xm )
{
    if ( xm >= 0.5 && xm <= 1.0 )
    {
        return 2.0;
    }
    return 1.0;
}		

void InitField( std::vector<float>& q, std::vector<float>& xcoor )
{
	int ni = xcoor.size();
	for ( int i = 0; i < ni; ++ i )
	{
		float fm = SquareFun( xcoor[i] );
		q[i] = fm;
	}
}

int main(int argc, char** argv)
{
    FieldPara *fieldPara = new FieldPara();
	fieldPara->Init();
	std::vector<float> q, theory, xcoor;
	q.resize( fieldPara->nx );
	theory.resize( fieldPara->nx );
	xcoor.resize( fieldPara->nx );
	GenerateGrid( fieldPara->nx, 0, fieldPara->len, xcoor );
	InitField( q, xcoor );
	delete fieldPara;
    return 0;
}
