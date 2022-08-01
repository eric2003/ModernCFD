cmake_minimum_required ( VERSION 3.20 )

project ( testprj )

set ( PRJ_SRC_LIST )
set ( PRJ_LIBRARIES )
set ( PRJ_INCLUDE_DIRS )

find_package ( MPI )

message ( STATUS  "MPI_FOUND=${MPI_FOUND}" )
message ( STATUS  "MPI_CXX_INCLUDE_DIRS=${MPI_CXX_INCLUDE_DIRS}" )
message ( STATUS  "MPI_LIBRARIES=${MPI_LIBRARIES}" )
if ( MPI_FOUND )
	list ( APPEND PRJ_INCLUDE_DIRS ${MPI_CXX_INCLUDE_DIRS} )
	list ( APPEND PRJ_LIBRARIES ${MPI_LIBRARIES} )
endif ()


set ( PRJ_SRC_LIST main.cpp )

add_executable ( ${PROJECT_NAME} ${PRJ_SRC_LIST} )

target_include_directories ( ${PROJECT_NAME}
	PRIVATE
		${PRJ_INCLUDE_DIRS}
)

target_compile_features ( ${PROJECT_NAME} 
	PUBLIC 
		cxx_std_20
)

target_link_libraries ( ${PROJECT_NAME} 
    PRIVATE 
        ${PRJ_LIBRARIES}
)