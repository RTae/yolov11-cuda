cmake_minimum_required(VERSION 3.18)  # Use at least CMake 3.18
project(OpenCV_CUDA_TensorRT LANGUAGES CXX CUDA)

# Set C++ Standard
set(CMAKE_CXX_STANDARD 11)

# Enable CUDA
find_package(CUDA REQUIRED)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Set the CUDA architectures explicitly
set(CMAKE_CUDA_ARCHITECTURES 75)  # Replace with the architecture of your GPU

# Include TensorRT paths
set(TENSORRT_INCLUDE_DIR /usr/include/tensorrt)  # TensorRT headers
set(TENSORRT_LIB_DIR /usr/lib/x86_64-linux-gnu)  # TensorRT libraries

# Include OpenCV, CUDA, and TensorRT headers
include_directories(${OpenCV_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS} ${TENSORRT_INCLUDE_DIR})

# Add the executable
add_executable(main main.cu)

# Specify CUDA as the language for the target
set_target_properties(main PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Link TensorRT, OpenCV, and CUDA libraries
target_link_libraries(main
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
    nvinfer         # TensorRT core library
    nvonnxparser    # TensorRT ONNX parser (if applicable)
    cudart          # CUDA runtime
)
