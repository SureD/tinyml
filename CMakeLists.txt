cmake_minimum_required(VERSION 3.20)

project(tiny_test LANGUAGES CXX OBJCXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(tinyinfer STATIC
    infer/src/core.cc
    infer/src/llama.cc
    infer/src/metal_backend.mm
)

target_include_directories(tinyinfer
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/infer/include
)

target_compile_options(tinyinfer PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
    $<$<COMPILE_LANGUAGE:OBJCXX>:-Wall -Wextra -Wpedantic>
)

add_executable(tinyinfer_tests
    tests/cpp/test_core.cc
)

target_link_libraries(tinyinfer_tests PRIVATE tinyinfer)
target_compile_options(tinyinfer_tests PRIVATE -Wall -Wextra -Wpedantic)

enable_testing()
add_test(NAME tinyinfer_tests COMMAND tinyinfer_tests)
