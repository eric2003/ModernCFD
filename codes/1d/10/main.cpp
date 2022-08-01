#define _USE_MATH_DEFINES // for C++
#include <cmath>
#include <omp.h>
#include <iostream>
#include <vector>
#include "matplotlibcppModified.h"
#include "TimeSpan.h"

namespace plt = matplotlibcpp;

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

void Theory( float time, float c, std::vector<float>& theory, std::vector<float>& xcoor )
{
    int ni = xcoor.size();
    float xs = c * time;
    for ( int i = 0; i < ni; ++ i )
    {
        float xm = xcoor[i];
        float xm_new = xm - xs;
        float fm = SquareFun( xm_new );
        theory[i] = fm;
    }
}

void Boundary( std::vector<float> &q )
{
    int ni = q.size();
    float xleft = 0.0;
    float qleft = SquareFun( xleft );
    q[ 0 ] = qleft;
}

void SolveField( FieldPara *fieldPara, std::vector<float> &q, std::vector<float> &qn, std::vector<float> &xcoor )
{
    std::cout << " fieldPara->nt = " << fieldPara->nt << "\n";  
    for ( int n = 0; n < fieldPara->nt; ++ n )
    {
        std::cout << " iStep = " << n + 1 << " nStep = " << fieldPara->nt << "\n";
        Boundary( q );
        for ( int i = 0; i < fieldPara->nx; ++ i )
        {
            qn[i] = q[i];
        }

        //float cfl = fieldPara->c * fieldPara->dt / fieldPara->dx;
        for ( int i = 1; i < fieldPara->nx; ++ i )
        {
            q[i] = qn[i] - fieldPara->cfl * ( qn[i] - qn[i-1] );
        }
    }
}

void Visual( std::vector<float> &q, std::vector<float> &theory, std::vector<float> &xcoor )
{
    // Set the size of output image to 1200x780 pixels
    plt::figure_size(1200, 780);
    // Plot line from given x and y data. Color is selected automatically.
    plt::plot(xcoor, q, {{"label", "calc"}});
    plt::plot(xcoor, theory, {{"label", "theory"}});
    // Add graph title
    plt::title("1d convection");
    plt::xlabel("x");
    plt::ylabel("u");
    // Enable legend.
    plt::legend();
    
    // Save the image (file format is determined by the extension)
    plt::savefig("./result.png");
}

int main(int argc, char** argv)
{
    FieldPara *fieldPara = new FieldPara();
    fieldPara->Init();
    std::vector<float> q, qn, theory, xcoor;
    q.resize( fieldPara->nx );
    qn.resize( fieldPara->nx );
    theory.resize( fieldPara->nx );
    xcoor.resize( fieldPara->nx );
    GenerateGrid( fieldPara->nx, 0, fieldPara->len, xcoor );
    InitField( q, xcoor );
    Theory( fieldPara->time, fieldPara->c, theory, xcoor );
    TimeSpan ts;
    SolveField( fieldPara, q, qn, xcoor );
    ts.ShowTimeSpan();

    Visual( q, theory, xcoor );
    delete fieldPara;

    return 0;
}