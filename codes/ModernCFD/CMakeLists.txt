cmake_minimum_required ( VERSION 3.20 )

project ( ModernCFD )

set ( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/modules/" )

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
message(STATUS "Build directory: ${PROJECT_BINARY_DIR}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

set ( PRJ_COMPILE_FEATURES )
set ( PRJ_COMPILE_DEFINITIONS )
set ( PRJ_LIBRARIES )
set ( PRJ_INCLUDE_DIRS )

list ( APPEND PRJ_COMPILE_FEATURES cxx_std_20 )

find_package ( MPI )

message ( STATUS  "MPI_FOUND=${MPI_FOUND}" )
message ( STATUS  "MPI_CXX_INCLUDE_DIRS=${MPI_CXX_INCLUDE_DIRS}" )
message ( STATUS  "MPI_LIBRARIES=${MPI_LIBRARIES}" )
if ( MPI_FOUND )
    list ( APPEND PRJ_INCLUDE_DIRS ${MPI_CXX_INCLUDE_DIRS} )
    list ( APPEND PRJ_LIBRARIES ${MPI_LIBRARIES} )
endif ()

find_package ( CUDAToolkit )
if ( CUDAToolkit_FOUND )
    enable_language( CUDA )
    set ( CMAKE_CUDA_STANDARD 20 )
    set ( CMAKE_CUDA_ARCHITECTURES "60;72;75" )
endif ()
message ( STATUS "CMAKE_CUDA_STANDARD=${CMAKE_CUDA_STANDARD}" )
find_package ( OpenMP REQUIRED )
#message ( STATUS  "OpenMP_FOUND=${OpenMP_FOUND}" )
list ( APPEND PRJ_LIBRARIES OpenMP::OpenMP_CXX )

set ( CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=${OpenMP_CXX_FLAGS}" )

find_package(Python3 COMPONENTS Interpreter Development)

message ( STATUS "Python3_FOUND = ${Python3_FOUND} " )
message ( STATUS "Python3_INCLUDE_DIRS = ${Python3_INCLUDE_DIRS} " )
message ( STATUS "Python3_LIBRARIES = ${Python3_LIBRARIES} " )

if ( ${Python3_FOUND} )
    #include_directories(${Python3_INCLUDE_DIRS})
else()
    message ( FATAL_ERROR "Python3 not found, please install it." )
endif()

find_package(NumPy)
if ( ${PYTHON_NUMPY_FOUND} )
    list ( APPEND PRJ_INCLUDE_DIRS ${PYTHON_NUMPY_INCLUDE_DIR} )
else()
    message(WARNING "Python3 NumPy not found, proceeding with -DWITHOUT_NUMPY."
    " Some functions might not work.")
    list ( APPEND PRJ_COMPILE_DEF WITHOUT_NUMPY )
endif()

list ( APPEND PRJ_INCLUDE_DIRS ${Python3_INCLUDE_DIRS} )
list ( APPEND PRJ_LIBRARIES ${Python3_LIBRARIES} )

message ( STATUS  "MSVC=${MSVC}" )
if ( MSVC )
    set_property( DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${PROJECT_NAME} )
endif()

option ( CFD_WITH_CUDA "Build Modern CFD(OneFLOW) with CUDA support" ON )
option ( CFD_WITH_OPENMP "Build Modern CFD(OneFLOW) with OPENMP support" ON )
option ( CFD_WITH_MPI "Build Modern CFD(OneFLOW) with MPI support" ON )

message ( STATUS  "CFD_WITH_CUDA=${CFD_WITH_CUDA}" )
message ( STATUS  "CFD_WITH_OPENMP=${CFD_WITH_OPENMP}" )
message ( STATUS  "CFD_WITH_MPI=${CFD_WITH_MPI}" )

if ( CFD_WITH_CUDA )
    list ( APPEND PRJ_COMPILE_DEFINITIONS ENABLE_CUDA )
endif ()

if ( CFD_WITH_OPENMP )
    list ( APPEND PRJ_COMPILE_DEFINITIONS ENABLE_OPENMP )
endif ()

if ( CFD_WITH_MPI )
    list ( APPEND PRJ_COMPILE_DEFINITIONS ENABLE_MPI )
endif ()

set( HEADER_LIST 
    Grid.h
    Geom.h
    Cmpi.h
    CfdPara.h
    Simu.h
    Solver.h
)

set( SRC_LIST 
    main.cpp
    Grid.cpp
    Geom.cpp
    Cmpi.cpp
    CfdPara.cpp
    Simu.cpp
    Solver.cu
)

add_executable ( ${PROJECT_NAME}
    ${HEADER_LIST}
    ${SRC_LIST}
)

set_target_properties( ${PROJECT_NAME}
    PROPERTIES 
        POSITION_INDEPENDENT_CODE ON
)

set_target_properties ( ${PROJECT_NAME}
    PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

target_include_directories ( ${PROJECT_NAME}
    PRIVATE
        ${PRJ_INCLUDE_DIRS}
)

target_link_libraries( ${PROJECT_NAME} 
    PRIVATE 
        ${PRJ_LIBRARIES}
)

target_compile_features ( ${PROJECT_NAME} 
    PRIVATE 
        ${PRJ_COMPILE_FEATURES}
)

target_compile_definitions ( ${PROJECT_NAME}
    PRIVATE
       ${PRJ_COMPILE_DEFINITIONS} 
)