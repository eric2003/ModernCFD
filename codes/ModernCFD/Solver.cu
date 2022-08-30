#include "Solver.h"
#include <string>
#include <set>
#include <map>
#include <fstream>
#include "Cmpi.h"
#include "Grid.h"
#include "Geom.h"
#include "CfdPara.h"
#include <omp.h>
#include <cuda_runtime.h>
#include "matplotlibcppModified.h"
namespace plt = matplotlibcpp;

float SquareFun( float xm )
{
    if ( xm >= 0.5 && xm <= 1.0 )
    {
        return 2.0;
    }
    return 1.0;
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

void Visual( float * q, float * xcoor, int ni, const std::string & fileName )
{
    std::vector<float> qv{ q + 1, q + ni };
    std::vector<float> xv{ xcoor + 1, xcoor + ni };
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

void Visual( std::vector<float> & q, std::vector<float> & theory, std::vector<float> & x,  const std::string & fileName )
{
    // Set the size of output image to 1200x780 pixels
    plt::figure_size(1200, 780);
    // Plot line from given x and y data. Color is selected automatically.
    plt::plot( x, q, { {"label", "OneFLOW"}, {"marker", "o" } } );
    plt::plot(x, theory, {{"label", "theory"}});
    // Add graph title
    plt::title("1d convection");
    plt::xlabel("x");
    plt::ylabel("u");
    // Enable legend.
    plt::legend();

    // Save the image (file format is determined by the extension)
    plt::savefig( fileName.c_str() );
}

Solver::Solver()
{
#ifdef PRJ_ENABLE_CUDA
    cudaGetDeviceCount( &Cmpi::num_gpus );
    if ( Cmpi::num_gpus < 1 ) {
        std::printf("no CUDA capable devices were detected\n");
        //std::exit(1);
    }
#endif
    std::printf("number of host CPUs:\t%d\n", omp_get_num_procs());
    std::printf("number of CUDA devices:\t%d\n", Cmpi::num_gpus);

#ifdef PRJ_ENABLE_CUDA
    for ( int i = 0; i < Cmpi::num_gpus; ++ i )
    {
        cudaDeviceProp dprop;
        cudaGetDeviceProperties( &dprop, i);
        std::printf("   %d: %s\n", i, dprop.name);
    }

    std::printf("---------------------------\n");
#endif

    int nCpuThreads = 8;
    omp_set_num_threads( nCpuThreads );
#ifdef ENABLE_OPENMP
#pragma omp parallel
#endif
    {
        unsigned int cpu_thread_id = omp_get_thread_num();
        unsigned int num_cpu_threads = omp_get_num_threads();
        //std::printf( "Solver::Solver() CPU thread %d (of %d)\n", cpu_thread_id, num_cpu_threads );
    }
}

Solver::~Solver()
{
    ;
}

void Solver::Init()
{
}

void Solver::Run( CfdPara * cfd_para, Geom * geom )
{
    this->CfdSolve( cfd_para, geom );
}

void Solver::AllocateField( Geom * geom )
{
    this->q = new float[ geom->ni_total ];
    this->qn = new float[ geom->ni_total ];
    this->timestep = new float[ geom->ni_total ];
}

void Solver::DeallocateField( Geom * geom )
{
    delete [] this->q;
    delete [] this->qn;
    delete [] this->timestep;
}

void Solver::InitField( CfdPara * cfd_para, Geom * geom )
{
    if ( cfd_para->irestart == 0 )
    {
        this->SetInflowField( cfd_para, geom );
    }
    else
    {
        this->ReadField( cfd_para, geom );
    }
}

void Solver::SetInflowField( CfdPara * cfd_para, Geom * geom )
{
    for ( int i = 0; i < geom->ni_total; ++ i )
    {
        float fm = SquareFun( geom->xcoor[ i ] );
        this->q[ i ] = fm;
    }
}

void Solver::ReadField( CfdPara * cfd_para, Geom * geom )
{
    for ( int i = 0; i < geom->ni_total; ++ i )
    {
        float fm = SquareFun( geom->xcoor[ i ] );
        this->q[ i ] = fm;
    }
}

void Solver::CfdSolve( CfdPara * cfd_para, Geom * geom )
{
    this->AllocateField( geom );
    this->InitField( cfd_para, geom );
    this->SolveField( cfd_para, geom );
    this->SaveField( cfd_para, geom );
    this->Visualize( cfd_para, geom );
    this->DeallocateField( geom );
}

void Solver::Timestep( CfdPara * cfd_para, Geom * geom )
{
    for ( int i = 0; i < geom->ni_total; ++ i )
    {
        this->timestep[ i ] = geom->ds[ i ] * cfd_para->cfl / cfd_para->cspeed;
    }
}

#ifdef PRJ_ENABLE_CUDA
__global__ void GpuCfdCopyVector( float *a, const float *b, int numElements )
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if ( i < numElements )
    {
        a[i] = b[i];
    }
}

__global__ void GpuCfdScalarUpdate( float * q, const float * qn, float c, const float * timestep, const float * ds, int ni )
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if ( i < ni + 1 && i > 0 )
    {
        float cfl = c * timestep[ i ] / ds[ i ];
        q[ i ] = qn[ i ] - cfl * ( qn[ i ] - qn[ i - 1 ] );
    }
}
#endif


void CfdCopyVector( float * a, float * b, int ni )
{
#ifdef PRJ_ENABLE_CUDA
    std::size_t nSize = ni * sizeof(float);

    float * dev_a;
    float * dev_b;
    cudaMalloc((void **)&dev_a, nSize);
    cudaMalloc((void **)&dev_b, nSize);

    cudaMemcpy(dev_a, a, nSize, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, nSize, cudaMemcpyHostToDevice);

    int block_size = 256;
    int grid_size = ( ni + block_size - 1 ) / block_size;
    dim3 grid_dim( grid_size );
    dim3 block_dim( block_size );  // 256 threads per block

    GpuCfdCopyVector<<<grid_dim, block_dim>>>( dev_a, dev_b, ni );
    cudaDeviceSynchronize();
    cudaMemcpy(a, dev_a, nSize, cudaMemcpyDeviceToHost);
    cudaFree(dev_a);
    cudaFree(dev_b);
#endif
}

void CfdCopyVectorSerial( float * a, float * b, int ni )
{
    for ( int i = 0; i < ni; ++ i )
    {
        a[ i ] = b[ i ];
    }
}

void CfdScalarUpdate( float * q, float * qn, float c, float * timestep, float * ds, int ni )
{
#ifdef PRJ_ENABLE_CUDA
    float * dev_q;
    float * dev_qn;
    float * dev_timestep;
    float * dev_ds;
    int nElem = ni + 2;
    std::size_t nSize = nElem * sizeof(float);

    cudaMalloc((void **)&dev_qn, nSize);
    cudaMalloc((void **)&dev_q, nSize);
    cudaMalloc((void **)&dev_timestep, nSize);
    cudaMalloc((void **)&dev_ds, nSize);

    cudaMemcpy(dev_qn, qn, nSize, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_q, q, nSize, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_ds, ds, nSize, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_timestep, timestep, nSize, cudaMemcpyHostToDevice);

    int block_size = 256;
    int grid_size = ( nElem + block_size - 1 ) / block_size;
    dim3 grid_dim( grid_size );
    dim3 block_dim( block_size );  // 256 threads per block

    //std::printf("Solver::SolveField CUDA kernel launch with %d blocks of %d threads\n", grid_size, block_size);
    GpuCfdScalarUpdate<<<grid_dim, block_dim>>>(dev_q, dev_qn, c, dev_timestep, dev_ds, ni);
    cudaDeviceSynchronize();
    cudaMemcpy(q, dev_q, nSize, cudaMemcpyDeviceToHost);
    cudaFree(dev_q);
    cudaFree(dev_qn);
    cudaFree(dev_timestep);
    cudaFree(dev_ds);
#endif
}

void CfdScalarUpdateSerial( float * q, float * qn, float c, float * timestep, float * ds, int ni )
{
    for ( int i = 1; i < ni + 1; ++ i )
    {
        float cfl = c * timestep[ i ] / ds[ i ];
        q[ i ] = qn[ i ] - cfl * ( qn[ i ] - qn[ i - 1 ] );
    }
}

void Solver::SolveField( CfdPara * cfd_para, Geom * geom )
{
    //for ( int i = 0; i < geom->ni_total; ++ i )
    //{
    //    qn[ i ] = q[ i ];
    //}
    for ( int n = 0; n < cfd_para->nt; ++ n )
    {
        if ( geom->zoneId == 0 )
        {
            std::printf( " iStep = %d, nStep = %d \n", n + 1, cfd_para->nt );
        }

        this->Boundary( q, geom );
        this->Timestep( cfd_para, geom );

        //int nCpuThreads = 4;
        int nCpuThreads = 1;
        omp_set_num_threads( nCpuThreads );
        #pragma omp parallel
        {
            int cpu_thread_id = omp_get_thread_num();
            int num_cpu_threads = omp_get_num_threads();
            int gpu_id = -1;
            cudaSetDevice( cpu_thread_id % Cmpi::num_gpus );
            cudaGetDevice( &gpu_id );
            //std::printf("Solver::SolveField CPU process %d (of %d) CPU thread %d (of %d) uses CUDA device %d\n", \
            //    Cmpi::pid, Cmpi::nproc, cpu_thread_id, num_cpu_threads, gpu_id);

            CfdCopyVector( qn, q, geom->ni_total );
            //CfdCopyVectorSerial( qn, q, geom->ni_total );
        }
        CfdScalarUpdate(this->q, this->qn, cfd_para->cspeed, this->timestep, geom->ds, geom->ni );
        //CfdScalarUpdateSerial(this->q, this->qn, cfd_para->cspeed, this->timestep, geom->ds, geom->ni );
    }
}

void Solver::Boundary( float * q, Geom * geom )
{
    BoundarySolver * bcSolver = geom->bcSolver;
    //physical boundary
    int nBFace = bcSolver->GetNBFace();
    //std::printf(" Boundary zoneID = %d nBFace = %d\n", bcSolver->zoneId, nBFace);
    for ( int iface = 0; iface < nBFace; ++ iface )
    {
        int bctype = bcSolver->bctypes[ iface ];
        int ghostcell_id = bcSolver->bc_ghostcells[ iface ];
        int bc_faceid = bcSolver->bc_faceids[ iface ];
        if ( bctype == BCInterface ) continue;
        if ( bctype == BCInflow )
        {
            float xm = geom->xcoor[ ghostcell_id ];
            q[ ghostcell_id ] = SquareFun( xm );
        }
        else if ( bctype == BCOutflow )
        {
            q[ ghostcell_id ] = q[ bc_faceid ];
        }
    }

    this->BoundaryInterface( q, geom );
}

void Solver::BoundaryInterface( float * q, Geom * geom )
{
    BoundarySolver * bcSolver = geom->bcSolver;
    int nIFace = bcSolver->GetNIFace();
    //std::printf( " BoundaryInterface nIFace = %d\n", nIFace );
    InterfaceSolver * interfaceSolver = bcSolver->interfaceSolver;
    interfaceSolver->SwapData( q );
    for ( int iface = 0; iface < nIFace; ++ iface )
    {
        int ghostcell_id = interfaceSolver->interface_ghost_cells[ iface ];
        //interfaceSolver->ShowInfo( iface );
        float bcvalue = interfaceSolver->GetBcValue( iface );
        q[ ghostcell_id ] = bcvalue;
    }
}

void Solver::SaveField( CfdPara * cfd_para, Geom * geom )
{
    char buffer[ 50 ];
    std::sprintf( buffer, "./flow%d.dat", geom->zoneId );
    std::fstream file;
    file.open( buffer, std::fstream::out | std::fstream::binary );

    int ni_total = geom->ni_total;
    file.write(reinterpret_cast<char *>(&ni_total), sizeof(int) );
    file.write(reinterpret_cast<char *>(this->q), ni_total * sizeof(float) );
    file.close();
}

void Solver::Visualize( CfdPara * cfd_para, Geom * geom )
{
    char buffer[ 50 ];
    std::sprintf( buffer, "./cfd%d.png", geom->zoneId );
    Visual( this->q, geom->xcoor, geom->ni, buffer );

    std::vector<float> q_global;
    std::vector<float> x_global;
    int root = 0;
    int tag = 0;
    if ( geom->zoneId != 0 )
    {
        MPI_Send( this->q, geom->ni_total, MPI_FLOAT, root, tag, MPI_COMM_WORLD );
    }
    else
    {
        std::vector<std::vector<float>> qvec( Cmpi::nproc );
        for ( int ip = 1; ip < Cmpi::nproc; ++ ip )
        {
            int ni_tmp = Geom_t::zonenis[ ip ];
            int ni_total_tmp = ni_tmp + Geom_t::ni_ghost;

            qvec[ ip ].resize( ni_total_tmp );
        }
        qvec[ 0 ].insert( qvec[ 0 ].end(), this->q, this->q + geom->ni_total );

        for ( int ip = 1; ip < Cmpi::nproc; ++ ip )
        {
            int ni_tmp = Geom_t::zonenis[ ip ];
            int ni_total_tmp = ni_tmp + Geom_t::ni_ghost;
            MPI_Recv( qvec[ ip ].data(), ni_total_tmp, MPI_FLOAT, ip, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE );
        }

        for ( int ip = 0; ip < Cmpi::nproc; ++ ip )
        {
            if ( ip == 0 )
            {
                q_global.insert( q_global.end(), qvec[ ip ].begin() + 1, qvec[ ip ].end() - 1 );
            }
            else
            {
                q_global.insert( q_global.end(), qvec[ ip ].begin() + 2, qvec[ ip ].end() - 1 );
            }
        }
        x_global.insert( x_global.end(), geom->xcoor_global + 1, geom->xcoor_global + Geom_t::ni_global + 1 );
        std::vector<float> theory;
        theory.resize( x_global.size() );
        Theory( cfd_para->simu_time, cfd_para->cspeed, theory, x_global );
        //Visual( q_global, theory, x_global, "./cfd.png" );
        Visual( q_global, theory, x_global, "./cfd.pdf" );
    }
}

