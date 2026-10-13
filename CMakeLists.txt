cmake_minimum_required(VERSION 3.20)

# Objective-C++ is the Apple Metal path. Linux configures with CXX only;
# metal_backend.mm is plain C++ when the Metal headers are not present.
project(tiny_test LANGUAGES CXX)
if(APPLE)
    enable_language(OBJCXX)
endif()

include(FetchContent)

FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3
    GIT_SHALLOW TRUE
)

FetchContent_MakeAvailable(nlohmann_json)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# metal_backend.mm is Objective-C++ only when Metal headers exist. On other
# hosts the same file is a C++ stub; compiling it as ObjC++ still emits
# __gnu_objc_personality_v0, which the Linux linker does not provide.
if(NOT APPLE)
    set_source_files_properties(
        infer/src/metal_backend.mm
        PROPERTIES LANGUAGE CXX)
endif()

add_library(tinyinfer STATIC
    infer/src/cpu_backend.cc
    infer/src/core.cc
    infer/src/embedder.cc
    infer/src/embedder_loader.cc
    infer/src/llama.cc
    infer/src/model_loader.cc
    infer/src/metal_backend.mm
)

target_include_directories(tinyinfer
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/infer/include
)

target_link_libraries(tinyinfer PRIVATE nlohmann_json::nlohmann_json)
if(APPLE)
    target_link_libraries(tinyinfer PRIVATE "-framework Metal" "-framework Foundation")
endif()

target_compile_options(tinyinfer PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
    $<$<COMPILE_LANGUAGE:OBJCXX>:-Wall -Wextra -Wpedantic>
)

add_executable(tinyinfer_tests
    tests/cpp/test_core.cc
)

target_link_libraries(tinyinfer_tests PRIVATE tinyinfer)
target_compile_options(tinyinfer_tests PRIVATE -Wall -Wextra -Wpedantic)

add_executable(load_checkpoint_smoke
    tests/cpp/load_checkpoint_smoke.cc
)

target_link_libraries(load_checkpoint_smoke PRIVATE tinyinfer)
target_compile_options(load_checkpoint_smoke PRIVATE -Wall -Wextra -Wpedantic)

add_executable(run_token_ids_smoke
    tests/cpp/run_token_ids_smoke.cc
)

target_link_libraries(run_token_ids_smoke PRIVATE tinyinfer)
target_compile_options(run_token_ids_smoke PRIVATE -Wall -Wextra -Wpedantic)

add_executable(bench_infer
    tests/cpp/bench_infer.cc
)

target_link_libraries(bench_infer PRIVATE tinyinfer)
target_compile_options(bench_infer PRIVATE -Wall -Wextra -Wpedantic)

add_executable(embedder_tests
    tests/cpp/test_embedder.cc
)

target_link_libraries(embedder_tests PRIVATE tinyinfer)
target_compile_options(embedder_tests PRIVATE -Wall -Wextra -Wpedantic)
target_compile_definitions(embedder_tests PRIVATE
    TINYINFER_EMBEDDER_FIXTURE_DIR="${CMAKE_SOURCE_DIR}/tests/data"
)

enable_testing()
add_test(NAME tinyinfer_tests COMMAND tinyinfer_tests)
add_test(NAME embedder_tests COMMAND embedder_tests)
