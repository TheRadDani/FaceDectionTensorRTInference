cmake_minimum_required(VERSION 3.20)
project(FaceDetector CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
cmake_policy(SET CMP0146 NEW)

# Find CUDA using modern approach
find_package(CUDAToolkit REQUIRED)

# Find TensorRT
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    HINTS /usr/include /usr/local/cuda/include
    PATH_SUFFIXES tensorrt)

find_library(TENSORRT_LIBRARY nvinfer
    HINTS /usr/lib /usr/local/cuda/lib64
    PATH_SUFFIXES x86_64-linux-gnu)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Verify dependencies
if(NOT TENSORRT_INCLUDE_DIR)
    message(FATAL_ERROR "TensorRT headers not found. Install with: sudo apt install libnvinfer-dev")
endif()

if(NOT TENSORRT_LIBRARY)
    message(FATAL_ERROR "TensorRT library not found. Install with: sudo apt install libnvinfer8")
endif()

message(STATUS "TensorRT include: ${TENSORRT_INCLUDE_DIR}")
message(STATUS "TensorRT library: ${TENSORRT_LIBRARY}")
message(STATUS "OpenCV: ${OpenCV_DIR}")

# Include directories
include_directories(${TENSORRT_INCLUDE_DIR})
include_directories(${OpenCV_INCLUDE_DIRS})

# Create executable
add_executable(face_detector face_detector.cu)

# Set CUDA architecture for maximum performance
set_target_properties(face_detector PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "75;80;86;89"
)

# Compiler optimizations
target_compile_options(face_detector PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -ffast-math>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math>
)

# Link libraries
target_link_libraries(face_detector PRIVATE
    ${TENSORRT_LIBRARY}
    CUDA::cudart
    ${OpenCV_LIBRARIES}
)

# Copy engine and image to build directory
configure_file(face-detector.engine face-detector.engine COPYONLY)
configure_file(lenna.png lenna.png COPYONLY)