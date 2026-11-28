
cmake_minimum_required(VERSION 3.16)

project(ParallelKMeans LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(include)

# Find OpenMP
find_package(OpenMP REQUIRED)

#Sequential implementation

add_executable(kmeans
    src/main.cpp
    src/kmeans.cpp
    src/dataset.cpp
)

target_compile_options(kmeans PRIVATE
    -Wall
    -Wextra
    -O2
)

#OpenMP implementation

add_executable(kmeans_openmp
    src/main.cpp
    src/kmeans_openmp.cpp
    src/dataset.cpp
)

target_compile_options(kmeans_openmp PRIVATE
    -Wall
    -Wextra
    -O2
)

target_link_libraries(kmeans_openmp
    PRIVATE OpenMP::OpenMP_CXX
)
