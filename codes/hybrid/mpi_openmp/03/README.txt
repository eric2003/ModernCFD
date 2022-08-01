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

PS D:\work\cfd_work\ModernCFD\codes\hybrid\mpi_openmp\02\build> mpiexec -n 4 .\Release\testprj.exe 2
Hybrid: Hello from thread 0 out of 1 from process 1 out of 4 on DELL
Hybrid: Hello from thread 0 out of 1 from process 2 out of 4 on DELL
Hybrid: Hello from thread 0 out of 1 from process 0 out of 4 on DELL
Hybrid: Hello from thread 0 out of 1 from process 3 out of 4 on DELL