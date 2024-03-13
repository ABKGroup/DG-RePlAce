# FFT CMAKE Compile Settings
project(fft LANGUAGES CXX)
cmake_minimum_required(VERSION 3.1)

set(FFT_HOME ${CMAKE_CURRENT_SOURCE_DIR})
set(FFT_SRC fftsg.cpp fftsg2d.cpp)
set(FFT_HEADERS fft.h)

add_library(fft STATIC ${FFT_SRC})

set_target_properties( fft
  PROPERTIES
  ARCHIVE_OUTPUT_DIRECTORY ${FFT_HOME}/lib
  LIBRARY_OUTPUT_DIRECTORY ${FFT_HOME}/lib
  RUNTIME_OUTPUT_DIRECTORY ${FFT_HOME}/lib
)

install(TARGETS fft DESTINATION lib)
