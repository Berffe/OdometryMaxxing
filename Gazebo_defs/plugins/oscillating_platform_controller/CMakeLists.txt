############################################################################
#
#   Copyright (c) 2025 PX4 Development Team. All rights reserved.
#
############################################################################

project(OscillatingPlatformController)

add_library(${PROJECT_NAME} SHARED
	MovingPlatformController.cpp
)

target_link_libraries(${PROJECT_NAME}
	PUBLIC ${GZ_PLUGIN_TARGET}
	PUBLIC ${GZ_SIM_TARGET}
	# GZ_SIM_TARGET should already bring in gz-transport/gz-msgs transitively
	# (gz-sim's own headers depend on them internally). These are only here
	# as a fallback if the build complains about missing gz/transport or
	# gz/msgs headers after adding the new pose publisher -- if the build
	# succeeds without them, or if these variable names aren't defined in
	# this PX4 build's CMake setup, remove the two lines below; an undefined
	# CMake variable here just expands to nothing, so leaving them is
	# harmless either way.
	PUBLIC ${GZ_TRANSPORT_TARGET}
	PUBLIC ${GZ_MSGS_TARGET}
)

target_include_directories(${PROJECT_NAME}
	PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
	PUBLIC ${CMAKE_CURRENT_BINARY_DIR}
)

if (NOT CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
	install(TARGETS ${PROJECT_NAME} LIBRARY DESTINATION lib/px4_gz_plugins)
endif()