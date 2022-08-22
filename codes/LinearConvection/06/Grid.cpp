#include "Grid.h"
#include <string>
#include "matplotlibcppModified.h"

namespace plt = matplotlibcpp;

GlobalInterface::GlobalInterface()
{
    ;
}

GlobalInterface::~GlobalInterface()
{
    ;
}

Interface::Interface()
{
    ;
}

Interface::~Interface()
{
    ;
}

int Interface::GetInterfaceCell(int iface)
{
    //0 1
    //0 ni
    return cells[iface];
}

float Interface::GetInterfaceValue(int iface)
{
    return q[iface];
}

void GenerateGrid( int ni, float xmin, float xmax, float * xcoor )
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

void InitField( float * q, float * xcoor, int ni )
{
    for ( int i = 0; i < ni; ++ i )
    {
        float fm = SquareFun( xcoor[i] );
        q[i] = fm;
    }
}

void Visual( float * q, float * xcoor, int ni, const std::string & fileName )
{
    std::vector<float> qv{ q, q + ni };
    std::vector<float> xv{ xcoor, xcoor + ni };
    // Set the size of output image to 1200x780 pixels
    plt::figure_size(1200, 780);
    // Plot line from given x and y data. Color is selected automatically.
    plt::plot(xv, qv, {{"label", "calc"}});
    // Add graph title
    plt::title("1d convection");
    plt::xlabel("x");
    plt::ylabel("u");
    // Enable legend.
    plt::legend();
    
    // Save the image (file format is determined by the extension)
    plt::savefig( fileName.c_str() );
}

void CfdSimulation( int ni, float * xcoor )
{
    float * q = new float[ ni ];
    InitField( q, xcoor, ni );
    Visual( q, xcoor, ni, "./cfd.png" );
    delete [] q;
}

void BoundaryInterface( Interface* myinterface, float* q )
{
    //Interface boundary
    int nIFace = myinterface->GetNIFace();
    for (int iface = 0; iface < nIFace; ++iface)
    {
        int pt = myinterface->GetInterfaceCell(iface);
        q[pt] = myinterface->GetInterfaceValue(iface);
    }
}

void Boundary( float * q, float* xcoor, int *bctypes, int nBFace, int nIFace, Interface* myinterface )
{
    //physical boundary
    for ( int i = 0; i < nBFace; ++ i )
    {
        int bctype = bctypes[i];
        if (bctype == BCInterface ) continue;
        if (bctype == BCInflow)
        {
            float xm = xcoor[i];
            q[ i ] = SquareFun( xm );
        }
        else if (bctype == BCOutflow)
        {
            ;
        }
    }

    BoundaryInterface( myinterface, q );
    ////Interface boundary
    //for (int iface = 0; iface < nIFace; ++iface)
    //{
    //    int pt = myinterface->GetInterfaceCell( iface );
    //    q[pt] = myinterface->GetInterfaceValue( iface );
    //}
}

void SolveField( float * q, float * qn, int ni, int nt, float cfl, int nBFace, int nIFace, int * bctypes, float* xcoor, Interface* myinterface)
{
    for ( int n = 0; n < nt; ++ n )
    {
        //std::printf(" iStep = %d, nStep = %d \n", n + 1, nt);
        Boundary(q, xcoor, bctypes, nBFace, nIFace, myinterface);
        for ( int i = 0; i < ni; ++ i )
        {
            qn[i] = q[i];
        }

        for ( int i = 1; i < ni; ++ i )
        {
            q[i] = qn[i] - cfl * ( qn[i] - qn[i-1] );
        }
    }
}

void CfdSolve( int zoneId, int ni, int nt, float cfl, float * xcoor, Interface *myinterface )
{
    float * q = new float[ ni ];
    float * qn = new float[ ni ];
    InitField( q, xcoor, ni );
    int nBFace = 2;
    int* bctypes = new int[ nBFace ];
    int nIFace = 2;
    SolveField( q, qn, ni, nt, cfl, nBFace, nIFace, bctypes, xcoor, myinterface);
    char buffer[ 50 ];
    std::sprintf(buffer, "./cfd%d.png", zoneId );
    Visual( q, xcoor, ni, buffer );
    delete [] q;
    delete [] qn;
    delete [] bctypes;
}
