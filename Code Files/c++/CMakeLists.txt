cmake_minimum_required(VERSION 3.20)
project(hft_engine CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

add_compile_options(-Wall -Wextra -Wpedantic -O3 -march=native)

# no external dependencies: the engine reads a flat binary replay file

add_executable(hft_engine
    src/main.cpp
    src/order_book.cpp
    src/queue_position.cpp
    src/glt_asymptotic.cpp
    src/riccati_lut.cpp
    src/regime_switch.cpp
    src/reservation_price.cpp
    src/fill_simulator.cpp
    src/pnl_tracker.cpp
    src/strategy.cpp
    src/replay_reader.cpp
    src/latency_bench.cpp
    src/epoch_clock.cpp
    src/params_atomic.cpp
    src/funding_correction.cpp
    src/ema_drift.cpp
)

target_include_directories(hft_engine PRIVATE include)

# fill-core scenario runner, built from the queue and fill units only
add_executable(fill_core_runner
    src/queue_position.cpp
    src/fill_simulator.cpp
    tests/fill_core_runner.cpp
)
target_include_directories(fill_core_runner PRIVATE include)
