
cmake_minimum_required(VERSION 3.26 FATAL_ERROR)
project(sphorb)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS}
                    include)

add_executable (example1 example1.cpp
                         pfm.cpp
                         utility.cpp
                         detector.cpp
                         nonmax.cpp
                         SPHORB.cpp)

target_link_libraries (example1 ${OpenCV_LIBRARIES})

add_executable (example2 example2.cpp
                         pfm.cpp
                         utility.cpp
                         detector.cpp
                         nonmax.cpp
                         SPHORB.cpp)

target_link_libraries (example2 ${OpenCV_LIBRARIES})

add_executable (example3 example3.cpp
                         pfm.cpp
                         utility.cpp
                         detector.cpp
                         nonmax.cpp
                         SPHORB.cpp)

target_link_libraries (example3 ${OpenCV_LIBRARIES})