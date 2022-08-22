cmd.exe "/K" '"C:\Program Files (x86)\Intel\oneAPI\setvars.bat" && powershell'
cmake -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icx -DCMAKE_BUILD_TYPE=Release -GNinja ..
cmake --build .
mpiexec -n 4 .\testprj.exe

cmake .. -D CMAKE_BUILD_TYPE=Debug
cmake --build . --config Debug
cmake .. -D CMAKE_BUILD_TYPE=Release
cmake --build . --config Release

PS D:\work\cfd_work\ModernCFD\codes\LinearConvection\12\build> mpiexec -n 4 .\Release\testprj.exe
bcinfo......
local_face_map.size()=2
local_face_map.size()=2
physical_bctypes.size()=1
physical_bcids.size()=1
interface_bctypes.size()=1
interface_bcids.size()=1
point id=1 bctype=1, bcName=BCInflow
point id=11 bctype=-1, bcName=BCInterface

BoundarySolver::Init pt: zoneId = 0 nZones = 4
global face map :
1->1 (zone,cell,ghost)[(0,2,0),(-1,-1,-1)] bcType = 1
11->2 (zone,cell,ghost)[(0,10,12),(1,12,10)] bcType = -1
21->2 (zone,cell,ghost)[(1,20,22),(2,22,20)] bcType = -1
31->2 (zone,cell,ghost)[(2,30,32),(3,32,30)] bcType = -1
41->1 (zone,cell,ghost)[(3,40,42),(-1,-1,-1)] bcType = 2

local face map :
1 -> 1
11 -> 11

 dt = 0.025000, dx = 0.050000, nt = 1, ni_global = 41
Running on 4 nodes
print xcoor: process id = 0 nproc = 4
-0.050000 0.000000 0.050000 0.100000 0.150000 0.200000 0.250000 0.300000 0.350000 0.400000 0.450000 0.500000 0.550000
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
bcinfo......
local_face_map.size()=2
local_face_map.size()=2
physical_bctypes.size()=0
physical_bcids.size()=0
interface_bctypes.size()=2
interface_bcids.size()=2
point id=1 bctype=-1, bcName=BCInterface
point id=11 bctype=-1, bcName=BCInterface

BoundarySolver::Init pt: zoneId = 1 nZones = 4
global face map :
1->1 (zone,cell,ghost)[(0,2,0),(-1,-1,-1)] bcType = 1
11->2 (zone,cell,ghost)[(0,10,12),(1,12,10)] bcType = -1
21->2 (zone,cell,ghost)[(1,20,22),(2,22,20)] bcType = -1
31->2 (zone,cell,ghost)[(2,30,32),(3,32,30)] bcType = -1
41->1 (zone,cell,ghost)[(3,40,42),(-1,-1,-1)] bcType = 2

local face map :
1 -> 11
11 -> 21

print xcoor: process id = 1 nproc = 4
0.450000 0.500000 0.550000 0.600000 0.650000 0.700000 0.750000 0.800000 0.850000 0.900000 0.950000 1.000000 1.050000
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
bcinfo......
local_face_map.size()=2
local_face_map.size()=2
physical_bctypes.size()=0
physical_bcids.size()=0
interface_bctypes.size()=2
interface_bcids.size()=2
point id=1 bctype=-1, bcName=BCInterface
point id=11 bctype=-1, bcName=BCInterface

BoundarySolver::Init pt: zoneId = 2 nZones = 4
global face map :
1->1 (zone,cell,ghost)[(0,2,0),(-1,-1,-1)] bcType = 1
11->2 (zone,cell,ghost)[(0,10,12),(1,12,10)] bcType = -1
21->2 (zone,cell,ghost)[(1,20,22),(2,22,20)] bcType = -1
31->2 (zone,cell,ghost)[(2,30,32),(3,32,30)] bcType = -1
41->1 (zone,cell,ghost)[(3,40,42),(-1,-1,-1)] bcType = 2

local face map :
1 -> 21
11 -> 31

print xcoor: process id = 2 nproc = 4
0.950000 1.000000 1.050000 1.100000 1.150000 1.200000 1.250000 1.300000 1.350000 1.400000 1.450000 1.500000 1.550000
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
bcinfo......
local_face_map.size()=2
local_face_map.size()=2
physical_bctypes.size()=1
physical_bcids.size()=1
interface_bctypes.size()=1
interface_bcids.size()=1
point id=11 bctype=2, bcName=BCOutflow
point id=1 bctype=-1, bcName=BCInterface

BoundarySolver::Init pt: zoneId = 3 nZones = 4
global face map :
1->1 (zone,cell,ghost)[(0,2,0),(-1,-1,-1)] bcType = 1
11->2 (zone,cell,ghost)[(0,10,12),(1,12,10)] bcType = -1
21->2 (zone,cell,ghost)[(1,20,22),(2,22,20)] bcType = -1
31->2 (zone,cell,ghost)[(2,30,32),(3,32,30)] bcType = -1
41->1 (zone,cell,ghost)[(3,40,42),(-1,-1,-1)] bcType = 2

local face map :
1 -> 31
11 -> 41

print xcoor: process id = 3 nproc = 4
1.450000 1.500000 1.550000 1.600000 1.650000 1.700000 1.750000 1.800000 1.850000 1.900000 1.950000 2.000000 2.050000
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
 iStep = 1, nStep = 1
 Boundary zoneID = 0 nBFace = 2
 id = 1, bctype = 1
 id = 11, bctype = -1
Average of square roots is: 0.667239
 iStep = 1, nStep = 1
 Boundary zoneID = 3 nBFace = 2
 id = 11, bctype = 2
 id = 1, bctype = -1
 iStep = 1, nStep = 1
 Boundary zoneID = 1 nBFace = 2
 id = 1, bctype = -1
 id = 11, bctype = -1
 iStep = 1, nStep = 1
 Boundary zoneID = 2 nBFace = 2
 id = 1, bctype = -1
 id = 11, bctype = -1
PASSED