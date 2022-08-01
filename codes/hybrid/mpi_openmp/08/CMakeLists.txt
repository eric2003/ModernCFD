cmake_minimum_required ( VERSION 3.20 )

project ( testprj )

set ( PRJ_COMPILE_FEATURES )
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

add_executable ( ${PROJECT_NAME}
    main.cpp
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
