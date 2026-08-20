cmake_minimum_required(VERSION 3.20)

project(
  snnbaseExperiments
  VERSION 0.1.0
  DESCRIPTION "Reproducible experiments built on the snnbase spiking neural network library"
  LANGUAGES CXX
)

include(CTest)

option(
  SNNBASE_EXPERIMENTS_ENABLE_TEMPORAL
  "Build LibTorch temporal residual SNN experiments"
  ON
)
option(
  SNNBASE_EXPERIMENTS_ENABLE_GUI
  "Build optional native experiment GUIs when their dependencies are available"
  ON
)

option(
  SNNBASE_EXPERIMENTS_FETCH_SNNBASE
  "Fetch snnbase at an immutable revision instead of using a local checkout"
  OFF
)
set(
  SNNBASE_SNNBASE_GIT_REPOSITORY
  "https://github.com/deanhorak/snnbase.git"
  CACHE STRING
  "snnbase repository used when fetching the dependency"
)
set(
  SNNBASE_SNNBASE_GIT_TAG
  ""
  CACHE STRING
  "Immutable snnbase tag or commit used when fetching the dependency"
)
set(
  SNNBASE_EXPECTED_SNNBASE_REVISION
  ""
  CACHE STRING
  "Expected snnbase commit; empty disables the configure-time check"
)
option(
  SNNBASE_REQUIRE_CLEAN_SNNBASE
  "Fail configuration when the local snnbase checkout has uncommitted changes"
  OFF
)

find_package(Git QUIET)

if(SNNBASE_EXPERIMENTS_FETCH_SNNBASE)
  if(NOT SNNBASE_SNNBASE_GIT_TAG)
    message(
      FATAL_ERROR
      "SNNBASE_SNNBASE_GIT_TAG must be an immutable tag or commit when "
      "SNNBASE_EXPERIMENTS_FETCH_SNNBASE is enabled."
    )
  endif()
  include(FetchContent)
  FetchContent_Declare(
    snnbase_dependency
    GIT_REPOSITORY "${SNNBASE_SNNBASE_GIT_REPOSITORY}"
    GIT_TAG "${SNNBASE_SNNBASE_GIT_TAG}"
    GIT_SHALLOW FALSE
  )
  FetchContent_Populate(snnbase_dependency)
  set(SNNBASE_SOURCE_DIR "${snnbase_dependency_SOURCE_DIR}")
else()
  set(
    SNNBASE_SOURCE_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/../snnbase"
    CACHE PATH
    "Path to a local snnbase source checkout"
  )
endif()
set(
  SNNBASE_EXPERIMENTS_MNIST_DATA_DIR
  "${CMAKE_CURRENT_SOURCE_DIR}/data/mnist"
  CACHE PATH
  "Directory containing the uncompressed MNIST IDX files"
)
set(
  SNNBASE_EXPERIMENTS_EMNIST_DATA_DIR
  "${CMAKE_CURRENT_SOURCE_DIR}/data/emnist"
  CACHE PATH
  "Directory containing the uncompressed EMNIST IDX files"
)
set(
  SNNBASE_EXPERIMENTS_CIFAR10_DATA_DIR
  "${CMAKE_CURRENT_SOURCE_DIR}/data/cifar-10-batches-bin"
  CACHE PATH
  "Directory containing the CIFAR-10 binary batch files"
)

if(NOT EXISTS "${SNNBASE_SOURCE_DIR}/CMakeLists.txt")
  message(
    FATAL_ERROR
    "snnbase was not found at '${SNNBASE_SOURCE_DIR}'. "
    "Set SNNBASE_SOURCE_DIR to a snnbase source checkout."
  )
endif()

set(SNNBASE_SNNBASE_GIT_REVISION "unknown")
set(SNNBASE_SNNBASE_GIT_DIRTY 0)
set(SNNBASE_EXPERIMENTS_GIT_REVISION "unknown")
set(SNNBASE_EXPERIMENTS_GIT_DIRTY 0)
if(Git_FOUND)
  execute_process(
    COMMAND "${GIT_EXECUTABLE}" -C "${SNNBASE_SOURCE_DIR}" rev-parse HEAD
    OUTPUT_VARIABLE SNNBASE_SNNBASE_GIT_REVISION
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
  )
  execute_process(
    COMMAND "${GIT_EXECUTABLE}" -C "${SNNBASE_SOURCE_DIR}" status --porcelain --untracked-files=all
    OUTPUT_VARIABLE SNNBASE_SNNBASE_STATUS
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
  )
  if(SNNBASE_SNNBASE_STATUS)
    set(SNNBASE_SNNBASE_GIT_DIRTY 1)
  endif()
  execute_process(
    COMMAND "${GIT_EXECUTABLE}" -C "${CMAKE_CURRENT_SOURCE_DIR}" rev-parse HEAD
    OUTPUT_VARIABLE SNNBASE_EXPERIMENTS_GIT_REVISION
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
  )
  execute_process(
    COMMAND "${GIT_EXECUTABLE}" -C "${CMAKE_CURRENT_SOURCE_DIR}" status --porcelain --untracked-files=all
    OUTPUT_VARIABLE SNNBASE_EXPERIMENTS_STATUS
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
  )
  if(SNNBASE_EXPERIMENTS_STATUS)
    set(SNNBASE_EXPERIMENTS_GIT_DIRTY 1)
  endif()
endif()
if(SNNBASE_EXPECTED_SNNBASE_REVISION AND
   NOT SNNBASE_SNNBASE_GIT_REVISION STREQUAL SNNBASE_EXPECTED_SNNBASE_REVISION)
  message(
    FATAL_ERROR
    "snnbase revision mismatch: expected '${SNNBASE_EXPECTED_SNNBASE_REVISION}', "
    "found '${SNNBASE_SNNBASE_GIT_REVISION}'."
  )
endif()
if(SNNBASE_REQUIRE_CLEAN_SNNBASE AND SNNBASE_SNNBASE_GIT_DIRTY)
  message(
    FATAL_ERROR
    "snnbase checkout '${SNNBASE_SOURCE_DIR}' has uncommitted changes."
  )
endif()
message(
  STATUS
  "snnbase revision: ${SNNBASE_SNNBASE_GIT_REVISION} "
  "(dirty=${SNNBASE_SNNBASE_GIT_DIRTY})"
)

set(SNNBASE_BUILD_TESTS OFF CACHE BOOL "Build snnbase's own tests")
set(SNNBASE_BUILD_EXAMPLES OFF CACHE BOOL "Build snnbase's own examples")
set(
  SNNBASE_ENABLE_TORCH
  ${SNNBASE_EXPERIMENTS_ENABLE_TEMPORAL}
  CACHE BOOL "Build snnbase's temporal residual backend"
)
set(SNNBASE_SPIKE_EVENT_BITS 128 CACHE STRING "Spike-event bit width")
set(SNNBASE_HISTORY_LENGTH 256 CACHE STRING "Prototype queue depth per class")
set(SNNBASE_DECISION_APPROACH 1 CACHE STRING "snnbase decision algorithm")
add_subdirectory("${SNNBASE_SOURCE_DIR}" "${CMAKE_BINARY_DIR}/_deps/snnbase")

add_library(
  snnbase_experiments
  src/cifar10.cpp
  src/emnist.cpp
  src/mnist.cpp
  src/nmnist.cpp
  src/spaun.cpp
  src/spaun_benchmark_runner.cpp
  src/spaun_benchmarks.cpp
  src/spaun_visual.cpp
  src/spaun_vision_benchmark.cpp
  src/spaun_workspace.cpp
)
if(SNNBASE_EXPERIMENTS_ENABLE_TEMPORAL)
  target_sources(snnbase_experiments PRIVATE src/spiking_conv.cpp)
endif()
add_library(snnbase_experiments::snnbase_experiments ALIAS snnbase_experiments)
target_compile_features(snnbase_experiments PUBLIC cxx_std_20)
target_include_directories(
  snnbase_experiments
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_link_libraries(
  snnbase_experiments
  PUBLIC snnbase::snnbase
)
if(SNNBASE_EXPERIMENTS_ENABLE_TEMPORAL)
  target_link_libraries(snnbase_experiments PUBLIC snnbase::temporal)
endif()

add_executable(mnist_experiment experiments/mnist/main.cpp)
target_link_libraries(mnist_experiment PRIVATE snnbase_experiments::snnbase_experiments)
add_executable(emnist_battery experiments/emnist/main.cpp)
target_link_libraries(emnist_battery PRIVATE snnbase_experiments::snnbase_experiments)
add_executable(cifar10_experiment experiments/cifar10/main.cpp)
target_link_libraries(cifar10_experiment PRIVATE snnbase_experiments::snnbase_experiments)
add_executable(nmnist_experiment experiments/nmnist/main.cpp)
target_link_libraries(nmnist_experiment PRIVATE snnbase_experiments::snnbase_experiments)
add_executable(spaun_experiment experiments/spaun/main.cpp)
target_link_libraries(spaun_experiment PRIVATE snnbase_experiments::snnbase_experiments)
add_executable(spaun_benchmark experiments/spaun/benchmark_main.cpp)
target_link_libraries(
  spaun_benchmark
  PRIVATE snnbase_experiments::snnbase_experiments
)
add_executable(
  spaun_vision_benchmark
  experiments/spaun/vision_benchmark_main.cpp
)
target_link_libraries(
  spaun_vision_benchmark
  PRIVATE snnbase_experiments::snnbase_experiments
)

if(SNNBASE_EXPERIMENTS_ENABLE_GUI)
  find_package(OpenGL QUIET)
  find_package(GLUT QUIET)
  if(OpenGL_FOUND AND GLUT_FOUND)
    add_executable(spaun_gui experiments/spaun/gui_main.cpp)
    target_link_libraries(
      spaun_gui
      PRIVATE
        snnbase_experiments::snnbase_experiments
        OpenGL::GL
        GLUT::GLUT
    )
  else()
    message(STATUS "OpenGL/GLUT not found; spaun_gui disabled")
  endif()
endif()
if(SNNBASE_EXPERIMENTS_ENABLE_TEMPORAL)
  add_executable(emnist_deep experiments/emnist/deep_main.cpp)
  target_link_libraries(emnist_deep PRIVATE snnbase_experiments::snnbase_experiments)
  add_executable(cifar10_deep experiments/cifar10/deep_main.cpp)
  target_link_libraries(cifar10_deep PRIVATE snnbase_experiments::snnbase_experiments)
endif()
configure_file(
  cmake/project_config.hpp.in
  generated/snnbase_experiments/project_config.hpp
  @ONLY
)
target_include_directories(
  mnist_experiment
  PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated"
)
target_include_directories(
  emnist_battery
  PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated"
)
target_include_directories(
  cifar10_experiment
  PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated"
)
target_include_directories(
  spaun_experiment
  PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated"
)
target_include_directories(
  spaun_vision_benchmark
  PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated"
)
if(SNNBASE_EXPERIMENTS_ENABLE_TEMPORAL)
  target_include_directories(emnist_deep PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated")
  target_include_directories(cifar10_deep PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated")
endif()

if(MSVC)
  target_compile_options(snnbase_experiments PRIVATE /W4 /WX)
  target_compile_options(mnist_experiment PRIVATE /W4 /WX)
  target_compile_options(emnist_battery PRIVATE /W4 /WX)
  target_compile_options(cifar10_experiment PRIVATE /W4 /WX)
  target_compile_options(nmnist_experiment PRIVATE /W4 /WX)
  target_compile_options(spaun_experiment PRIVATE /W4 /WX)
  target_compile_options(spaun_benchmark PRIVATE /W4 /WX)
  target_compile_options(spaun_vision_benchmark PRIVATE /W4 /WX)
  if(TARGET spaun_gui)
    target_compile_options(spaun_gui PRIVATE /W4 /WX)
  endif()
  if(SNNBASE_EXPERIMENTS_ENABLE_TEMPORAL)
    target_compile_options(emnist_deep PRIVATE /W4 /WX)
    target_compile_options(cifar10_deep PRIVATE /W4 /WX)
  endif()
else()
  target_compile_options(snnbase_experiments PRIVATE -Wall -Wextra -Wpedantic -Werror)
  target_compile_options(mnist_experiment PRIVATE -Wall -Wextra -Wpedantic -Werror)
  target_compile_options(emnist_battery PRIVATE -Wall -Wextra -Wpedantic -Werror)
  target_compile_options(cifar10_experiment PRIVATE -Wall -Wextra -Wpedantic -Werror)
  target_compile_options(nmnist_experiment PRIVATE -Wall -Wextra -Wpedantic -Werror)
  target_compile_options(spaun_experiment PRIVATE -Wall -Wextra -Wpedantic -Werror)
  target_compile_options(
    spaun_benchmark
    PRIVATE -Wall -Wextra -Wpedantic -Werror
  )
  target_compile_options(
    spaun_vision_benchmark
    PRIVATE -Wall -Wextra -Wpedantic -Werror
  )
  if(TARGET spaun_gui)
    target_compile_options(spaun_gui PRIVATE -Wall -Wextra -Wpedantic -Werror)
  endif()
  if(SNNBASE_EXPERIMENTS_ENABLE_TEMPORAL)
    target_compile_options(emnist_deep PRIVATE -Wall -Wextra -Wpedantic -Werror)
    target_compile_options(cifar10_deep PRIVATE -Wall -Wextra -Wpedantic -Werror)
  endif()
endif()

if(BUILD_TESTING)
  add_executable(spaun_tests tests/spaun_tests.cpp)
  target_link_libraries(spaun_tests PRIVATE snnbase_experiments::snnbase_experiments)
  if(MSVC)
    target_compile_options(spaun_tests PRIVATE /W4 /WX)
  else()
    target_compile_options(spaun_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
  endif()
  add_test(NAME spaun_tests COMMAND spaun_tests)
  set_tests_properties(spaun_tests PROPERTIES LABELS "spaun;snn;cognitive")

  add_executable(spaun_benchmarks_tests tests/spaun_benchmarks_tests.cpp)
  target_link_libraries(
    spaun_benchmarks_tests
    PRIVATE snnbase_experiments::snnbase_experiments
  )
  if(MSVC)
    target_compile_options(spaun_benchmarks_tests PRIVATE /W4 /WX)
  else()
    target_compile_options(
      spaun_benchmarks_tests
      PRIVATE -Wall -Wextra -Wpedantic -Werror
    )
  endif()
  add_test(NAME spaun_benchmarks_tests COMMAND spaun_benchmarks_tests)
  set_tests_properties(
    spaun_benchmarks_tests
    PROPERTIES LABELS "spaun;benchmark;statistics"
  )

  add_executable(
    spaun_benchmark_runner_tests
    tests/spaun_benchmark_runner_tests.cpp
  )
  target_link_libraries(
    spaun_benchmark_runner_tests
    PRIVATE snnbase_experiments::snnbase_experiments
  )
  if(MSVC)
    target_compile_options(spaun_benchmark_runner_tests PRIVATE /W4 /WX)
  else()
    target_compile_options(
      spaun_benchmark_runner_tests
      PRIVATE -Wall -Wextra -Wpedantic -Werror
    )
  endif()
  add_test(
    NAME spaun_benchmark_runner_tests
    COMMAND spaun_benchmark_runner_tests
  )
  set_tests_properties(
    spaun_benchmark_runner_tests
    PROPERTIES LABELS "spaun;benchmark;integration"
  )

  add_executable(
    spaun_vision_benchmark_tests
    tests/spaun_vision_benchmark_tests.cpp
  )
  target_link_libraries(
    spaun_vision_benchmark_tests
    PRIVATE snnbase_experiments::snnbase_experiments
  )
  if(MSVC)
    target_compile_options(spaun_vision_benchmark_tests PRIVATE /W4 /WX)
  else()
    target_compile_options(
      spaun_vision_benchmark_tests
      PRIVATE -Wall -Wextra -Wpedantic -Werror
    )
  endif()
  add_test(
    NAME spaun_vision_benchmark_tests
    COMMAND spaun_vision_benchmark_tests
  )

  add_executable(spaun_visual_tests tests/spaun_visual_tests.cpp)
  target_link_libraries(
    spaun_visual_tests
    PRIVATE snnbase_experiments::snnbase_experiments
  )
  if(MSVC)
    target_compile_options(spaun_visual_tests PRIVATE /W4 /WX)
  else()
    target_compile_options(
      spaun_visual_tests
      PRIVATE -Wall -Wextra -Wpedantic -Werror
    )
  endif()
  add_test(NAME spaun_visual_tests COMMAND spaun_visual_tests)
  set_tests_properties(
    spaun_visual_tests
    PROPERTIES LABELS "spaun;visual;snn;causal"
  )

  add_executable(spaun_workspace_tests tests/spaun_workspace_tests.cpp)
  target_link_libraries(
    spaun_workspace_tests
    PRIVATE snnbase_experiments::snnbase_experiments
  )
  if(MSVC)
    target_compile_options(spaun_workspace_tests PRIVATE /W4 /WX /UNDEBUG)
  else()
    target_compile_options(
      spaun_workspace_tests
      PRIVATE -Wall -Wextra -Wpedantic -Werror -UNDEBUG
    )
  endif()
  add_test(NAME spaun_workspace_tests COMMAND spaun_workspace_tests)
  set_tests_properties(
    spaun_workspace_tests
    PROPERTIES LABELS "spaun;memory;semantic;snn;causal"
  )

  add_test(
    NAME spaun_task_battery
    COMMAND spaun_experiment --task all --minimum-accuracy 0.75
  )
  set_tests_properties(
    spaun_task_battery
    PROPERTIES LABELS "spaun;snn;cognitive;integration" TIMEOUT 120
  )

  if(TARGET spaun_gui)
    add_test(NAME spaun_gui_controller COMMAND spaun_gui --self-test)
    set_tests_properties(
      spaun_gui_controller
      PROPERTIES LABELS "spaun;gui;smoke" TIMEOUT 120
    )
  endif()

  add_executable(mnist_tests tests/mnist_tests.cpp)
  target_link_libraries(mnist_tests PRIVATE snnbase_experiments::snnbase_experiments)
  if(MSVC)
    target_compile_options(mnist_tests PRIVATE /W4 /WX)
  else()
    target_compile_options(mnist_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
  endif()
  add_test(NAME mnist_tests COMMAND mnist_tests)
  set_tests_properties(mnist_tests PROPERTIES LABELS "mnist;unit")

  add_executable(emnist_tests tests/emnist_tests.cpp)
  target_link_libraries(emnist_tests PRIVATE snnbase_experiments::snnbase_experiments)
  if(MSVC)
    target_compile_options(emnist_tests PRIVATE /W4 /WX)
  else()
    target_compile_options(emnist_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
  endif()
  add_test(NAME emnist_tests COMMAND emnist_tests)
  set_tests_properties(emnist_tests PROPERTIES LABELS "emnist;unit")

  add_executable(nmnist_tests tests/nmnist_tests.cpp)
  target_link_libraries(nmnist_tests PRIVATE snnbase_experiments::snnbase_experiments)
  if(MSVC)
    target_compile_options(nmnist_tests PRIVATE /W4 /WX)
  else()
    target_compile_options(nmnist_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
  endif()
  add_test(NAME nmnist_tests COMMAND nmnist_tests)
  set_tests_properties(nmnist_tests PROPERTIES LABELS "nmnist;event;unit")

  if(SNNBASE_EXPERIMENTS_ENABLE_TEMPORAL)
    add_executable(spiking_conv_tests tests/spiking_conv_tests.cpp)
    target_link_libraries(spiking_conv_tests PRIVATE snnbase_experiments::snnbase_experiments)
    if(MSVC)
      target_compile_options(spiking_conv_tests PRIVATE /W4 /WX)
    else()
      target_compile_options(spiking_conv_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
    endif()
    add_test(NAME spiking_conv_tests COMMAND spiking_conv_tests)
    set_tests_properties(spiking_conv_tests PROPERTIES LABELS "temporal;unit")
  endif()

  add_executable(cifar10_tests tests/cifar10_tests.cpp)
  target_link_libraries(cifar10_tests PRIVATE snnbase_experiments::snnbase_experiments)
  if(MSVC)
    target_compile_options(cifar10_tests PRIVATE /W4 /WX)
  else()
    target_compile_options(cifar10_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
  endif()
  add_test(NAME cifar10_tests COMMAND cifar10_tests)
  set_tests_properties(cifar10_tests PROPERTIES LABELS "cifar10;unit")

  set(_cifar10_dataset_complete TRUE)
  foreach(
    _cifar10_file
    data_batch_1.bin
    data_batch_2.bin
    data_batch_3.bin
    data_batch_4.bin
    data_batch_5.bin
    test_batch.bin
  )
    if(NOT EXISTS "${SNNBASE_EXPERIMENTS_CIFAR10_DATA_DIR}/${_cifar10_file}")
      set(_cifar10_dataset_complete FALSE)
    endif()
  endforeach()

  if(_cifar10_dataset_complete)
    add_test(
      NAME cifar10_dataset_smoke
      COMMAND
        cifar10_experiment
        --data-dir "${SNNBASE_EXPERIMENTS_CIFAR10_DATA_DIR}"
        --train-limit 1000
        --test-limit 200
        --epochs 1
        --channels 4
        --batch-size 16
        --time-steps 4
    )
    set_tests_properties(
      cifar10_dataset_smoke
      PROPERTIES LABELS "cifar10;smoke" TIMEOUT 120
    )
    if(SNNBASE_EXPERIMENTS_ENABLE_TEMPORAL)
      add_test(
        NAME cifar10_deep_smoke
        COMMAND
          cifar10_deep
          --data-dir "${SNNBASE_EXPERIMENTS_CIFAR10_DATA_DIR}"
          --epochs 1
          --width 4
          --train-limit 20
          --test-limit 10
      )
      set_tests_properties(
        cifar10_deep_smoke
        PROPERTIES LABELS "cifar10;smoke;deep-snn" TIMEOUT 120
      )
    endif()
  else()
    message(
      STATUS
      "CIFAR-10 data not found at ${SNNBASE_EXPERIMENTS_CIFAR10_DATA_DIR}; "
      "CIFAR-10 dataset smoke test disabled"
    )
  endif()

  set(_mnist_dataset_complete TRUE)
  foreach(
    _mnist_file
    train-images-idx3-ubyte
    train-labels-idx1-ubyte
    t10k-images-idx3-ubyte
    t10k-labels-idx1-ubyte
  )
    if(NOT EXISTS "${SNNBASE_EXPERIMENTS_MNIST_DATA_DIR}/${_mnist_file}")
      set(_mnist_dataset_complete FALSE)
    endif()
  endforeach()

  if(_mnist_dataset_complete)
    add_test(
      NAME mnist_dataset_smoke
      COMMAND
        mnist_experiment
        --data-dir "${SNNBASE_EXPERIMENTS_MNIST_DATA_DIR}"
        --train-limit 100
        --test-limit 20
    )
    add_test(
      NAME mnist_accuracy
      COMMAND
        mnist_experiment
        --data-dir "${SNNBASE_EXPERIMENTS_MNIST_DATA_DIR}"
        --minimum-accuracy 0.85
    )
    set_tests_properties(mnist_dataset_smoke PROPERTIES LABELS "mnist;smoke")
    set_tests_properties(
      mnist_accuracy
      PROPERTIES LABELS "mnist;accuracy" TIMEOUT 300
    )
  else()
    message(
      STATUS
      "MNIST data not found at ${SNNBASE_EXPERIMENTS_MNIST_DATA_DIR}; "
      "dataset smoke test disabled"
    )
  endif()

  set(_emnist_dataset_complete TRUE)
  set(_emnist_splits byclass bymerge balanced letters digits mnist)
  foreach(_emnist_split IN LISTS _emnist_splits)
    foreach(_emnist_partition train test)
      foreach(_emnist_kind images-idx3-ubyte labels-idx1-ubyte)
        if(
          NOT EXISTS
          "${SNNBASE_EXPERIMENTS_EMNIST_DATA_DIR}/emnist-${_emnist_split}-${_emnist_partition}-${_emnist_kind}"
        )
          set(_emnist_dataset_complete FALSE)
        endif()
      endforeach()
    endforeach()
  endforeach()

  if(_emnist_dataset_complete)
    set(_emnist_floors 0.66 0.70 0.66 0.73 0.92 0.92)
    foreach(_emnist_index RANGE 0 5)
      list(GET _emnist_splits ${_emnist_index} _emnist_split)
      list(GET _emnist_floors ${_emnist_index} _emnist_floor)
      add_test(
        NAME emnist_accuracy_${_emnist_split}
        COMMAND
          emnist_battery
          --data-dir "${SNNBASE_EXPERIMENTS_EMNIST_DATA_DIR}"
          --architecture structured
          --splits "${_emnist_split}"
          --minimum-accuracy "${_emnist_floor}"
      )
      set_tests_properties(
        emnist_accuracy_${_emnist_split}
        PROPERTIES LABELS "emnist;benchmark" TIMEOUT 420
      )
    endforeach()
    if(SNNBASE_EXPERIMENTS_ENABLE_TEMPORAL)
      add_test(
        NAME emnist_deep_mnist_accuracy
        COMMAND
          emnist_deep
          --data-dir "${SNNBASE_EXPERIMENTS_EMNIST_DATA_DIR}"
          --split mnist
          --epochs 5
          --minimum-accuracy 0.98
      )
      set_tests_properties(
        emnist_deep_mnist_accuracy
        PROPERTIES LABELS "emnist;benchmark;deep-snn" TIMEOUT 300
      )
      add_test(
        NAME emnist_deep_digits_accuracy
        COMMAND
          emnist_deep
          --data-dir "${SNNBASE_EXPERIMENTS_EMNIST_DATA_DIR}"
          --split digits
          --epochs 3
          --minimum-accuracy 0.985
      )
      set_tests_properties(
        emnist_deep_digits_accuracy
        PROPERTIES LABELS "emnist;benchmark;deep-snn" TIMEOUT 420
      )
    endif()
  else()
    message(
      STATUS
      "EMNIST data not found at ${SNNBASE_EXPERIMENTS_EMNIST_DATA_DIR}; "
      "EMNIST accuracy battery disabled"
    )
  endif()
endif()
