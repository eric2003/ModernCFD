cmd.exe "/K" '"C:\Program Files (x86)\Intel\oneAPI\setvars.bat" && powershell'
cmake -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icx -DCMAKE_BUILD_TYPE=Release -GNinja ..
cmake --build .
mpiexec -n 4 .\testprj.exe

cmake .. -D CMAKE_BUILD_TYPE=Debug
cmake --build . --config Debug
cmake .. -D CMAKE_BUILD_TYPE=Release
cmake --build . --config Release

mpiexec -n 4 .\Release\testprj.exe

PS D:\work\cfd_work\ModernCFD\codes\mpi\init\01\build> mpiexec -n 4 .\Release\testprj.exe
Hello, world!  I am 0 of 4(Microsoft MPI 10.1.12498.18, 27)
Hello, world!  I am 3 of 4(Microsoft MPI 10.1.12498.18, 27)
Hello, world!  I am 2 of 4(Microsoft MPI 10.1.12498.18, 27)
Hello, world!  I am 1 of 4(Microsoft MPI 10.1.12498.18, 27)