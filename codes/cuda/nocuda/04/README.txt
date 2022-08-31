cmd.exe "/K" '"C:\Program Files (x86)\Intel\oneAPI\setvars.bat" && powershell'
cmake -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icx -DCMAKE_BUILD_TYPE=Release -GNinja ..
cmake --build .
mpiexec -n 4 .\testprj.exe

cmake .. -D CMAKE_BUILD_TYPE=Debug
cmake --build . --config Debug
cmake .. -D CMAKE_BUILD_TYPE=Release
cmake --build . --config Release

cmake .. -D CMAKE_BUILD_TYPE=Release `
-D CFD_WITH_CUDA:BOOL=ON

cmake .. -D CMAKE_BUILD_TYPE=Release `
-D CFD_WITH_CUDA:BOOL=OFF

PS D:\work\cfd_work\ModernCFD\codes\cuda\nocuda\04\build> cmake .. -D CMAKE_BUILD_TYPE=Release -D CFD_WITH_CUDA:BOOL=OFF
-- Building for: Visual Studio 17 2022
-- Selecting Windows SDK version 10.0.19041.0 to target Windows 10.0.22000.
-- The C compiler identification is MSVC 19.33.31629.0
-- The CXX compiler identification is MSVC 19.33.31629.0
-- Detecting C compiler ABI info
-- Detecting C compiler ABI info - done
-- Check for working C compiler: C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.33.31629/bin/Hostx64/x64/cl.exe - skipped
-- Detecting C compile features
-- Detecting C compile features - done
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Check for working CXX compiler: C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.33.31629/bin/Hostx64/x64/cl.exe - skipped
-- Detecting CXX compile features
-- Detecting CXX compile features - done
-- Build directory: D:/work/cfd_work/ModernCFD/codes/cuda/nocuda/04/build
-- Build type: Release
-- CFD_WITH_CUDA=OFF
-- CFD_WITH_CUDA=OFF
-- Configuring done
CMake Error: CMake can not determine linker language for target: testprj
-- Generating done
CMake Generate step failed.  Build files cannot be regenerated correctly.
PS D:\work\cfd_work\ModernCFD\codes\cuda\nocuda\04\build> cmake --build . --config Release
MSBuild version 17.3.1+2badb37d1 for .NET Framework
C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Current\Bin\amd64\Microsoft.Common.CurrentVersion.targe
ts(2066,5): warning : The referenced project 'D:\work\cfd_work\ModernCFD\codes\cuda\nocuda\04\build\testprj.vcxproj' do
es not exist. [D:\work\cfd_work\ModernCFD\codes\cuda\nocuda\04\build\ALL_BUILD.vcxproj]
  Building Custom Rule D:/work/cfd_work/ModernCFD/codes/cuda/nocuda/04/CMakeLists.txt