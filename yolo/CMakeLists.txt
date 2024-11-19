project(yolo_deploy)

add_executable(${PROJECT_NAME}
    ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/yolov8.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/detect_service.cpp
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