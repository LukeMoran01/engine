cmake_minimum_required(VERSION 3.29)
project(engine)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# SDL3 path configuration
if(WIN32)
    set(SDL3_LIBRARY "${CMAKE_CURRENT_SOURCE_DIR}/external/SDL3/lib/x64/SDL3.lib")
endif()

set(CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/external/SDL3")
set(SDL3_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/SDL3/include")

set(GLM_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/glm")

# Your executable
add_executable(${PROJECT_NAME}
        src/main.cpp
        # Add other source files here
)

if(WIN32)
    # Add SDL3.dll to output directory
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_CURRENT_SOURCE_DIR}/external/SDL3/bin/SDL3.dll"
            $<TARGET_FILE_DIR:${PROJECT_NAME}>
    )
endif()

find_package(Vulkan REQUIRED)

# Link SDL3
target_link_libraries(${PROJECT_NAME} PRIVATE ${SDL3_LIBRARY} Vulkan::Vulkan)

# Include directories
target_include_directories(${PROJECT_NAME} PRIVATE
        ${SDL3_INCLUDE_DIR}
        Vulkan::Vulkan
        glm::glm
        "src"
)

message(STATUS "SDL3_DIR: ${SDL3_DIR}")
message(STATUS "SDL3_LIBRARY: ${SDL3_LIBRARY}")
message(STATUS "SDL3_INCLUDE_DIR: ${SDL3_INCLUDE_DIR}")
