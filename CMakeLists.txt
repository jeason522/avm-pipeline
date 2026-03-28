cmake_minimum_required(VERSION 3.10)
project(avm_pipeline LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

set(ORT_DIR $ENV{HOME}/onnxruntime-linux-x64-1.16.3)

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${ORT_DIR}/include
    ${CMAKE_SOURCE_DIR}/include
)

link_directories(${ORT_DIR}/lib)

# Full pipeline executable
add_executable(pipeline
    src/pipeline.cpp
    src/fisheye_calib.cpp
    src/extrinsic_calib.cpp
    src/undistort.cpp
    src/birdview_stitch.cpp
    src/detect.cpp
)
target_link_libraries(pipeline ${OpenCV_LIBS} onnxruntime)

# Standalone calibration tool
add_executable(calibrate src/fisheye_calib.cpp)
target_compile_definitions(calibrate PRIVATE STANDALONE_CALIB)
target_link_libraries(calibrate ${OpenCV_LIBS})
