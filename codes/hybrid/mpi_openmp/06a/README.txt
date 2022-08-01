cmd.exe "/K" '"C:\Program Files (x86)\Intel\oneAPI\setvars.bat" && powershell'
cmake -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icx -DCMAKE_BUILD_TYPE=Release -GNinja ..
cmake --build .
mpiexec -n 4 .\testprj.exe

cmake .. -D CMAKE_BUILD_TYPE=Debug
cmake --build . --config Debug
cmake .. -D CMAKE_BUILD_TYPE=Release
cmake --build . --config Release

mpiexec -n 4 .\Release\testprj.exe

cmake .. -D CMAKE_BUILD_TYPE=Release
cmake --build . --config Release
mpiexec -n 4 .\Release\testprj.exe 2

PS D:\work\cfd_work\ModernCFD\codes\hybrid\mpi_openmp\06a\build> mpiexec -n 2 .\Release\testprj.exe
OpenMP: Number of threads = 1
OpenMP: Number of local_nthreads = 4
I'm thread 0 in process 0
OpenMP: Number of local_nthreads = 4
I'm thread 1 in process 0
OpenMP: Number of local_nthreads = 4
I'm thread 2 in process 0
OpenMP: Number of local_nthreads = 4
I'm thread 3 in process 0
OpenMP: Number of threads = 1
OpenMP: Number of local_nthreads = 4
I'm thread 0 in process 1
OpenMP: Number of local_nthreads = 4
I'm thread 1 in process 1
OpenMP: Number of local_nthreads = 4
I'm thread 2 in process 1
OpenMP: Number of local_nthreads = 4
I'm thread 3 in process 1