project(yolo-bytetrack)

file(GLOB_RECURSE src ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
# message(${src})

add_executable(${PROJECT_NAME}
    ${src}
    ${CommonDir}/common.hpp
)

target_include_directories(${PROJECT_NAME}
    PUBLIC
    ${ALL_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(${PROJECT_NAME}
    PRIVATE
    ${ALL_LIBS}
)