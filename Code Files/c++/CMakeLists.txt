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
    main.cpp
    order_book.cpp
    queue_position.cpp
    glt_asymptotic.cpp
    riccati_lut.cpp
    regime_switch.cpp
    reservation_price.cpp
    fill_simulator.cpp
    pnl_tracker.cpp
    strategy.cpp
    replay_reader.cpp
    latency_bench.cpp
    epoch_clock.cpp
    params_atomic.cpp
    funding_correction.cpp
    ema_drift.cpp
)

target_include_directories(hft_engine PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# fill-core scenario runner, built from the queue and fill units only
add_executable(fill_core_runner
    queue_position.cpp
    fill_simulator.cpp
    fill_core_runner.cpp
)
target_include_directories(fill_core_runner PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
