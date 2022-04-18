cmake_minimum_required(VERSION 2.6)
project(PCL_RECON)

find_package(PCL QUIET)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

#file(GLOB native_srcs RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
#aux_source_directory(. native_srcs)
#add_executable("${native_srcs}" "${native_srcs}")
#target_link_libraries ("${file${native_srcs}}" ${PCL_LIBRARIES})
#install(TARGETS "${file${native_srcs}}" RUNTIME DESTINATION bin)

file(GLOB cppfile RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

foreach(cpp ${cppfile})
    MESSAGE(${cpp})
    STRING(REGEX REPLACE ".cpp" "o" cppname ${cpp})
    MESSAGE(${cppname})
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ../../../bin/)
    add_executable(${cppname} ${cpp})
    target_link_libraries(${cppname} ${PCL_LIBRARIES})
    install(TARGETS "${cppname}" RUNTIME DESTINATION bin)
endforeach()



