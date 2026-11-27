
cmake_minimum_required(VERSION 3.16)

project(ParallelKMeans LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include header files
include_directories(include)

# Sequential K-Means executable
add_executable(kmeans
    src/main.cpp
    src/kmeans.cpp
    src/dataset.cpp
)

# Enable compiler warnings
target_compile_options(kmeans PRIVATE
    -Wall
    -Wextra
    -O2
)
