cmake_minimum_required(VERSION 3.10)
project(nvidiaGPU-optimized-image-processing)

set(CMAKE_CXX_STANDARD 17)

enable_language(CUDA)
set(OpenCV_DIR "$ENV{HOME}/.local/opencv/lib64/cmake/opencv4")
find_package(OpenCV REQUIRED)

# Include directories
include_directories(
    ${PROJECT_SOURCE_DIR}/external  # cxxopts.hpp is here
    ${OpenCV_DIR}
)


set(SOURCES
    main.cpp
    parser.cpp
    image_process.cu
)

add_executable(GPU_app ${SOURCES})
set_target_properties(GPU_app PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(GPU_app ${OpenCV_LIBS})
# Link CUDA if using .cu files (optional)
# target_link_libraries(csc_app cuda)

# If you're using CUDA .cu files, you could do:
# add_executable(csc_app main.cpp parser.cpp cuda_kernels.cu)

