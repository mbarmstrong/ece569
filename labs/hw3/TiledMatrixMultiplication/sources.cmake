
add_lab("TiledMatrixMultiplication")
add_lab_solution("TiledMatrixMultiplication" ${CMAKE_CURRENT_LIST_DIR}/solution_template_with_timer_utility_without_wb_library.cu)
add_generator("TiledMatrixMultiplication" ${CMAKE_CURRENT_LIST_DIR}/dataset_generator.cpp)
