cmake_minimum_required(VERSION 3.12)

project(Model_Deploy)
enable_testing()
set(CMAKE_CXX_STANDARD 17)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(Function)

# CUDA
find_package(CUDA REQUIRED)
print_var(CUDA_LIBRARIES)
print_var(CUDA_INCLUDE_DIRS)
get_filename_component(CUDA_LIB_DIR ${CUDA_LIBRARIES} DIRECTORY)
print_var(CUDA_LIB_DIR)

# OpenCV
set(CMAKE_PREFIX_PATH "F:/ThirdPart/OpenCV" ${CMAKE_PREFIX_PATH})
find_package(OpenCV REQUIRED)
if (${OpenCV_VERSION} VERSION_GREATER_EQUAL 4.7.0)
    message(STATUS "Build with -DBATCHED_NMS")
    add_definitions(-DBATCHED_NMS)
endif ()
print_var(OpenCV_LIBS)
print_var(OpenCV_LIBRARIES)
print_var(OpenCV_INCLUDE_DIRS)

# TensorRT
list(APPEND CMAKE_PREFIX_PATH "F:/ThirdPart/TensorRT-8.4.3.1" ${CMAKE_PREFIX_PATH})
find_package(TensorRT REQUIRED)
print_var(TensorRT_LIBRARIES)
print_var(TensorRT_INCLUDE_DIRS)
print_var(TensorRT_LIB_DIR)
if (TensorRT_VERSION_MAJOR GREATER_EQUAL 10)
    message(STATUS "Build with -DTRT_10")
    add_definitions(-DTRT_10)
endif ()

# Eigen
list(APPEND CMAKE_PREFIX_PATH "F:/ThirdPart/Eigen3" ${CMAKE_PREFIX_PATH})
find_package(Eigen3 REQUIRED)

set(CommonDir ${CMAKE_SOURCE_DIR}/common)

list(APPEND ALL_INCLUDE_DIRS
        ${CUDA_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        ${TensorRT_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CommonDir}
)

list(APPEND ALL_LIBS
        ${CUDA_LIBRARIES}
        ${OpenCV_LIBRARIES}
        ${TensorRT_LIBRARIES}
        Eigen3::Eigen
)

list(APPEND ALL_LIB_DIRS
        ${CUDA_LIB_DIR}
        ${TensorRT_LIB_DIR}
)

print_var(ALL_INCLUDE_DIRS)
print_var(ALL_LIBS)
print_var(ALL_LIB_DIRS)



# 添加子目录
add_subdirectory(yolo)
add_subdirectory(zero-dce)
add_subdirectory(concurrency_infer)
add_subdirectory(sahi)
# add_subdirectory(yolo-seg)
add_subdirectory(yolo-bytetrack)