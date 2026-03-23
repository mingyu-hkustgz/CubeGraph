cmake_minimum_required(VERSION 3.15)
project(FilteredVectorKNN CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/third_party/)

find_package(OpenMP)

if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
else()
    message(FATAL_ERROR "no OpenMP supprot")
endif()


find_package(Boost COMPONENTS REQUIRED program_options filesystem)
message(STATUS "Boost_INCLUDE_DIRS: ${Boost_INCLUDE_DIRS}")
if (Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIR})
else()
    message(FATAL_ERROR "no Boost supprot")
endif()

# Main library
add_library(filtered_knn INTERFACE)
target_include_directories(filtered_knn INTERFACE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/third_party/hnswlib
)

add_definitions(-std=c++17 -Ofast -march=core-avx2 -mavx512f -fpic -fopenmp -ftree-vectorize -fexceptions)

add_subdirectory(src)


