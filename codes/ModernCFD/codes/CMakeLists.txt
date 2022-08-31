set ( SRC_ROOT ${CMAKE_CURRENT_SOURCE_DIR} )
list ( APPEND PRJ_INCLUDE_DIRS ${SRC_ROOT} )

message ( STATUS "SRC_ROOT=${SRC_ROOT}" )

BuildSlnTree( ${SRC_ROOT} PRJ_HEADER_LIST PRJ_SRC_LIST PRJ_LOCAL_INCLUDE_DIRS )
list ( APPEND PRJ_INCLUDE_DIRS ${PRJ_LOCAL_INCLUDE_DIRS} )

#set( HEADER_LIST 
#    ${SRC_ROOT}/Grid.h
#    ${SRC_ROOT}/Geom.h
#    ${SRC_ROOT}/Cmpi.h
#    ${SRC_ROOT}/CfdPara.h
#    ${SRC_ROOT}/Simu.h
#    ${SRC_ROOT}/Solver.h
#    ${SRC_ROOT}/SolverDetail.h
#    ${SRC_ROOT}/SolverDetailCpu.h
#    ${SRC_ROOT}/SolverDetailCuda.h
#)

#set( SRC_LIST 
#    ${SRC_ROOT}/main.cpp
#    ${SRC_ROOT}/Grid.cpp
#    ${SRC_ROOT}/Geom.cpp
#    ${SRC_ROOT}/Cmpi.cpp
#    ${SRC_ROOT}/CfdPara.cpp
#    ${SRC_ROOT}/Simu.cpp
#    ${SRC_ROOT}/Solver.cpp
#    ${SRC_ROOT}/SolverDetail.cpp
#    ${SRC_ROOT}/SolverDetailCpu.cpp
#    ${SRC_ROOT}/SolverDetailCuda.cu
#)

add_executable ( ${PROJECT_NAME}
    ${PRJ_HEADER_LIST}
    ${PRJ_SRC_LIST}
)

set_target_properties( ${PROJECT_NAME}
    PROPERTIES 
        POSITION_INDEPENDENT_CODE ON
)

if ( CUDAToolkit_FOUND )
set_target_properties ( ${PROJECT_NAME}
    PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
endif()

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