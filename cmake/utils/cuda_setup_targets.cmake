include(${CMAKE_CURRENT_LIST_DIR}/cuda_auto_arch.cmake)

function(cuda_setup_targets)
  set(oneValueArgs CUDA_MIN_ARCH)
  set(multiValueArgs TARGETS FEATURES OPTIONS LIBS INCLUDES SCOPE)
  cmake_parse_arguments(ARG "" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  if(NOT ARG_TARGETS)
    message(FATAL_ERROR "cuda_setup_targets: TARGETS argument is required")
  endif()

  if(NOT ARG_SCOPE)
    set(ARG_SCOPE PRIVATE)
  endif()

  if(NOT ARG_SCOPE MATCHES "^(PRIVATE|PUBLIC|INTERFACE)$")
    message(FATAL_ERROR
            "cuda_setup_targets: SCOPE must be PRIVATE, PUBLIC or INTERFACE")
  endif()

  foreach(target_name IN LISTS ARG_TARGETS)
    if(NOT TARGET ${target_name})
      message(WARNING
              "cuda_setup_targets: Target '${target_name}' does not exist")
      continue()
    endif()

    if(ARG_CUDA_MIN_ARCH)
      set_target_arch_auto(${target_name} ${ARG_CUDA_MIN_ARCH})
    endif()

    if(ARG_INCLUDES)
      target_include_directories(${target_name} ${ARG_SCOPE} ${ARG_INCLUDES})
    endif()

    if(ARG_FEATURES)
      target_compile_features(${target_name} ${ARG_SCOPE} ${ARG_FEATURES})
    endif()

    if(ARG_OPTIONS)
      target_compile_options(${target_name} ${ARG_SCOPE} ${ARG_OPTIONS})
    endif()

    if(ARG_LIBS)
      target_link_libraries(${target_name} ${ARG_SCOPE} ${ARG_LIBS})
    endif()

    message(STATUS
            "Configured CUDA target: ${target_name} (scope: ${ARG_SCOPE})")
  endforeach()
endfunction()
