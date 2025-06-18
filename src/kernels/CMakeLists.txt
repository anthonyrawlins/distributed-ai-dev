cmake_minimum_required(VERSION 3.16)
project(ROCmOptimizations LANGUAGES CXX HIP)

# ROCm setup
find_package(hip REQUIRED)

# Compiler flags for RDNA3/CDNA3 optimization
set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} --offload-arch=gfx1100")
set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -O3 -ffast-math")

# Include directories
include_directories(${HIP_INCLUDE_DIRS})

# Attention optimization library (simplified)
add_library(attention_optimization SHARED
    attention_optimization_simplified.hip
    memory_optimization.hip
)

target_link_libraries(attention_optimization 
    hip::host
)

# Test executable (simplified)
add_executable(test_kernels_simple
    test_kernels_simple.cpp
)

target_link_libraries(test_kernels_simple 
    attention_optimization
    hip::host
)

# Installation
install(TARGETS attention_optimization 
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(FILES 
    attention_optimization.hip
    memory_optimization.hip
    DESTINATION include
)