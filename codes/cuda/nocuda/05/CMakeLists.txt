cmake_minimum_required ( VERSION 3.20 )

project ( testprj )

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
message(STATUS "Build directory: ${PROJECT_BINARY_DIR}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

set ( PRJ_COMPILE_FEATURES )
set ( PRJ_LIBRARIES )
set ( PRJ_INCLUDE_DIRS )
set ( PRJ_COMPILE_DEFINITIONS )

list ( APPEND PRJ_COMPILE_FEATURES cxx_std_20 )

option ( CFD_WITH_CUDA "Build Modern CFD(OneFLOW) with CUDA support" ON )
message ( STATUS  "CFD_WITH_CUDA=${CFD_WITH_CUDA}" )

find_package ( CUDAToolkit )
if ( CUDAToolkit_FOUND )
    enable_language( CUDA )
    set ( CMAKE_CUDA_STANDARD 20 )
    set ( CMAKE_CUDA_ARCHITECTURES "60;72;75" )
    message ( STATUS "CMAKE_CUDA_STANDARD=${CMAKE_CUDA_STANDARD}" )
endif ()

if ( CUDAToolkit_FOUND AND CFD_WITH_CUDA )
    list ( APPEND PRJ_COMPILE_DEFINITIONS ENABLE_CUDA )
endif ()

message ( STATUS "CFD_WITH_CUDA=${CFD_WITH_CUDA}" )

set( HEADER_LIST 
    sub.h
)

set( SRC_LIST 
    main.cpp
    sub.cu
)

add_executable ( ${PROJECT_NAME}
    ${HEADER_LIST}
    ${SRC_LIST}
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
