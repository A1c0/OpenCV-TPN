cmake_minimum_required(VERSION 3.13)
project(OpenCV_TPN)

set(CMAKE_CXX_STANDARD 14)

add_executable(OpenCV_TPN main.cpp)

find_package( OpenCV REQUIRED )

FILE(COPY video/ DESTINATION ./video)

include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(OpenCV_TPN ${OpenCV_LIBS})