cmd.exe "/K" '"C:\Program Files (x86)\Intel\oneAPI\setvars.bat" && powershell'
cmake -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icx -DCMAKE_BUILD_TYPE=Release -GNinja ..
cmake --build .
mpiexec -n 4 .\testprj.exe

cmake .. -D CMAKE_BUILD_TYPE=Debug
cmake --build . --config Debug
cmake .. -D CMAKE_BUILD_TYPE=Release
cmake --build . --config Release

PS D:\work\cfd_work\ModernCFD\codes\LinearConvection\05\build> mpiexec -n 4 .\Release\testprj.exe
 dt = 0.025000, dx = 0.050000, nt = 25, ni_global = 41
Running on 4 nodes
print xcoor: process id = 1 nproc = 4
0.500000 0.550000 0.600000 0.650000 0.700000 0.750000 0.800000 0.850000 0.900000 0.950000 1.000000
Starting addConstantGpu...

number of host CPUs:    16
number of CUDA devices: 1
   0: NVIDIA GeForce RTX 2060
---------------------------
CPU thread 0 (of 2) uses CUDA device 0
block_size=128
grid_size=32
CPU thread 1 (of 2) uses CUDA device 0
block_size=128
grid_size=32
CHECK PASSED!
sumNode = 1.70836e+06 process id = 1 nproc = 4
print xcoor: process id = 3 nproc = 4
1.500000 1.550000 1.600000 1.650000 1.700000 1.750000 1.800000 1.850000 1.900000 1.950000 2.000000
Starting addConstantGpu...

number of host CPUs:    16
number of CUDA devices: 1
   0: NVIDIA GeForce RTX 2060
---------------------------
CPU thread 0 (of 2) uses CUDA device 0
block_size=128
grid_size=32
CPU thread 1 (of 2) uses CUDA device 0
block_size=128
grid_size=32
CHECK PASSED!
sumNode = 1.70732e+06 process id = 3 nproc = 4
print xcoor: process id = 2 nproc = 4
1.000000 1.050000 1.100000 1.150000 1.200000 1.250000 1.300000 1.350000 1.400000 1.450000 1.500000
Starting addConstantGpu...

number of host CPUs:    16
number of CUDA devices: 1
   0: NVIDIA GeForce RTX 2060
---------------------------
CPU thread 0 (of 2) uses CUDA device 0
block_size=128
grid_size=32
CPU thread 1 (of 2) uses CUDA device 0
block_size=128
grid_size=32
CHECK PASSED!
sumNode = 1.70802e+06 process id = 2 nproc = 4
print boundary pt: process id = 0 nproc = 4
0 10 20 30 40
print xcoor: process id = 0 nproc = 4
0.000000 0.050000 0.100000 0.150000 0.200000 0.250000 0.300000 0.350000 0.400000 0.450000 0.500000
Starting addConstantGpu...

number of host CPUs:    16
number of CUDA devices: 1
   0: NVIDIA GeForce RTX 2060
---------------------------
CPU thread 0 (of 2) uses CUDA device 0
block_size=128
grid_size=32
CPU thread 1 (of 2) uses CUDA device 0
block_size=128
grid_size=32
CHECK PASSED!
sumNode = 1.70882e+06 process id = 0 nproc = 4
Average of square roots is: 0.667239
PASSED