add_executable(index_cube test_index_cube.cpp)

add_executable(search_cube test_search_cube.cpp)

add_executable(link_adjacent_cube test_link_adjacent_cube.cpp)

add_executable(cross_cube_search test_cross_cube_search.cpp)

add_executable(bench_hierarchical_cube bench_hierarchical_cube.cpp)

add_executable(bench_hierarchical_ball bench_hierarchical_ball.cpp)

add_executable(bench_hierarchical_polygon bench_hierarchical_polygon.cpp)

add_executable(bench_post_filtering bench_post_filtering.cpp)

add_executable(bench_complex_filter bench_complex_filter.cpp)

add_executable(test_predetermined_search test_predetermined_search.cpp)

add_executable(bench_rtree_partition bench_rtree_partition.cpp)

add_executable(bench_kdtree_partition bench_kdtree_partition.cpp)

option(BUILD_PYTHON_BINDINGS "Build Python bindings" ON)
if(BUILD_PYTHON_BINDINGS)
	execute_process(
		COMMAND python3 -m pybind11 --cmakedir
		OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
		OUTPUT_STRIP_TRAILING_WHITESPACE
	)
	if(PYBIND11_CMAKE_DIR)
		list(APPEND CMAKE_PREFIX_PATH ${PYBIND11_CMAKE_DIR})
		find_package(pybind11 CONFIG REQUIRED)

		pybind11_add_module(rtree_partition_py python_bindings.cpp)
		target_include_directories(rtree_partition_py PRIVATE
			${CMAKE_SOURCE_DIR}/include
			${CMAKE_SOURCE_DIR}/third_party
			${CMAKE_SOURCE_DIR}/third_party/hnswlib
		)
	else()
		message(WARNING "pybind11 cmake directory not found; skipping Python bindings")
	endif()
endif()
