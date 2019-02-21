set(CMAKE_POSITION_INDEPENDENT_CODE ON
  CACHE BOOL "Position independent code" FORCE)

# Collect the names of the source files
file(GLOB SOURCE_FILES "${CMAKE_CURRENT_LIST_DIR}/libtike/*.[ch]")
list(FILTER SOURCE_FILES EXCLUDE REGEX "raytrace*")
foreach(file ${SOURCE_FILES})
  message(STATUS "Added source file: ${file}")
endforeach(file)

add_library(tike SHARED ${SOURCE_FILES})

set_target_properties(tike PROPERTIES PREFIX "lib" C_STANDARD 11)

if(WIN32)
  install(TARGETS tike RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/src/tike)
else()
  install(TARGETS tike LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/src/tike)
endif()
