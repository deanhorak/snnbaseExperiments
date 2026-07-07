cmake_minimum_required(VERSION 3.20)

project(
  snnbaseExperiments
  VERSION 0.1.0
  DESCRIPTION "Reproducible experiments built on the snnbase spiking neural network library"
  LANGUAGES CXX
)

include(CTest)

set(
  SNNBASE_SOURCE_DIR
  "${CMAKE_CURRENT_SOURCE_DIR}/../snnbase"
  CACHE PATH
  "Path to a local snnbase source checkout"
)
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

if(NOT EXISTS "${SNNBASE_SOURCE_DIR}/CMakeLists.txt")
  message(
    FATAL_ERROR
    "snnbase was not found at '${SNNBASE_SOURCE_DIR}'. "
    "Set SNNBASE_SOURCE_DIR to a snnbase source checkout."
  )
endif()

set(SNNBASE_BUILD_TESTS OFF CACHE BOOL "Build snnbase's own tests")
set(SNNBASE_BUILD_EXAMPLES OFF CACHE BOOL "Build snnbase's own examples")
set(SNNBASE_SPIKE_EVENT_BITS 128 CACHE STRING "Spike-event bit width")
set(SNNBASE_HISTORY_LENGTH 256 CACHE STRING "Prototype queue depth per class")
set(SNNBASE_DECISION_APPROACH 1 CACHE STRING "snnbase decision algorithm")
add_subdirectory("${SNNBASE_SOURCE_DIR}" "${CMAKE_BINARY_DIR}/_deps/snnbase")

add_library(
  snnbase_experiments
  src/emnist.cpp
  src/mnist.cpp
  src/spiking_conv.cpp
)
add_library(snnbase_experiments::snnbase_experiments ALIAS snnbase_experiments)
target_compile_features(snnbase_experiments PUBLIC cxx_std_20)
target_include_directories(
  snnbase_experiments
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_link_libraries(snnbase_experiments PUBLIC snnbase::snnbase)

add_executable(mnist_experiment experiments/mnist/main.cpp)
target_link_libraries(mnist_experiment PRIVATE snnbase_experiments::snnbase_experiments)
add_executable(emnist_battery experiments/emnist/main.cpp)
target_link_libraries(emnist_battery PRIVATE snnbase_experiments::snnbase_experiments)
add_executable(emnist_deep experiments/emnist/deep_main.cpp)
target_link_libraries(emnist_deep PRIVATE snnbase_experiments::snnbase_experiments)
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
  emnist_deep
  PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated"
)

if(MSVC)
  target_compile_options(snnbase_experiments PRIVATE /W4 /WX)
  target_compile_options(mnist_experiment PRIVATE /W4 /WX)
  target_compile_options(emnist_battery PRIVATE /W4 /WX)
  target_compile_options(emnist_deep PRIVATE /W4 /WX)
else()
  target_compile_options(snnbase_experiments PRIVATE -Wall -Wextra -Wpedantic -Werror)
  target_compile_options(mnist_experiment PRIVATE -Wall -Wextra -Wpedantic -Werror)
  target_compile_options(emnist_battery PRIVATE -Wall -Wextra -Wpedantic -Werror)
  target_compile_options(emnist_deep PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()

if(BUILD_TESTING)
  add_executable(mnist_tests tests/mnist_tests.cpp)
  target_link_libraries(mnist_tests PRIVATE snnbase_experiments::snnbase_experiments)
  if(MSVC)
    target_compile_options(mnist_tests PRIVATE /W4 /WX)
  else()
    target_compile_options(mnist_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
  endif()
  add_test(NAME mnist_tests COMMAND mnist_tests)

  add_executable(emnist_tests tests/emnist_tests.cpp)
  target_link_libraries(emnist_tests PRIVATE snnbase_experiments::snnbase_experiments)
  if(MSVC)
    target_compile_options(emnist_tests PRIVATE /W4 /WX)
  else()
    target_compile_options(emnist_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
  endif()
  add_test(NAME emnist_tests COMMAND emnist_tests)

  add_executable(spiking_conv_tests tests/spiking_conv_tests.cpp)
  target_link_libraries(spiking_conv_tests PRIVATE snnbase_experiments::snnbase_experiments)
  if(MSVC)
    target_compile_options(spiking_conv_tests PRIVATE /W4 /WX)
  else()
    target_compile_options(spiking_conv_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
  endif()
  add_test(NAME spiking_conv_tests COMMAND spiking_conv_tests)

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
    set(_emnist_floors 0.63 0.68 0.67 0.73 0.91 0.90)
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
  else()
    message(
      STATUS
      "EMNIST data not found at ${SNNBASE_EXPERIMENTS_EMNIST_DATA_DIR}; "
      "EMNIST accuracy battery disabled"
    )
  endif()
endif()
