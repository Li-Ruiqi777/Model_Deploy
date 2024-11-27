project(concurrency_infer)

add_executable(${PROJECT_NAME}
    ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/zero-dce.cpp
    ${CommonDir}/common.hpp
)

target_include_directories(${PROJECT_NAME}
    PUBLIC
    ${ALL_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/
)

target_link_libraries(${PROJECT_NAME}
    PRIVATE
    ${ALL_LIBS}
)