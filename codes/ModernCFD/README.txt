cmd.exe "/K" '"C:\Program Files (x86)\Intel\oneAPI\setvars.bat" && powershell'
cmake -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icx -DCMAKE_BUILD_TYPE=Release -GNinja ..
cmake --build .
mpiexec -n 4 .\testprj.exe

cmake .. -D CMAKE_BUILD_TYPE=Debug
cmake --build . --config Debug
cmake .. -D CMAKE_BUILD_TYPE=Release
cmake --build . --config Release

cmake .. -D CMAKE_BUILD_TYPE=Release -D TEST
cmake .. -D CMAKE_BUILD_TYPE=Release -D TEST:BOOL=ON

cmake .. -D CMAKE_BUILD_TYPE=Release -D CFD_WITH_CUDA:BOOL=OFF

int grid_ni = ( Geom_t::ni_global + nZones - 1 ) / nZones;
( grid_ni - 1  ) * nZones + 1 = ( Geom_t::ni_global + nZones - 1 ) + 1

400001;
4000001;
 iStep = 2499990, nStep = 2500000
 iStep = 2499991, nStep = 2500000
 i
zone ni----------------------
1000001 1000001 1000001 1000001
 right_bcface_id =4000001
print xcoor: process id = 2 Cmpi::nproc = 4

number of host CPUs:    16
number of CUDA devices: 1
   0: NVIDIA GeForce RTX 2060
---------------------------
zone ni----------------------
1000001 1000001 1000001 1000001
 right_bcface_id =4000001
print xcoor: process id = 3 Cmpi::nproc = 4

number of host CPUs:    16
number of CUDA devices: 1
   0: NVIDIA GeForce RTX 2060
---------------------------
zone ni----------------------
1000001 1000001 1000001 1000001
 right_bcface_id =4000001
print xcoor: process id = 1 Cmpi::nproc = 4

number of host CPUs:    16
number of CUDA devices: 1
   0: NVIDIA GeForce RTX 2060
---------------------------
Step = 2499992, nStep = 2500000
 iStep = 2499993, nStep = 2500000
 iStep = 2499994, nStep = 2500000
 iStep = 2499995, nStep = 2500000
 iStep = 2499996, nStep = 2500000
 iStep = 2499997, nStep = 2500000
 iStep = 2499998, nStep = 2500000
 iStep = 2499999, nStep = 2500000
 iStep = 2500000, nStep = 2500000