project(
    'Vulkan-Project',
    'c',
    'cpp',
    version: '1.0.0',
    default_options: ['cpp_std=c++20', 'default_library=static'],
)

cpp_compiler = meson.get_compiler('cpp')

add_project_arguments('-DGLM_FORCE_DEFAULT_ALIGNED_GENTYPES', '-DGLM_FORCE_DEPTH_ZERO_TO_ONE', language: 'cpp')

if cpp_compiler.get_id() == 'gcc'
    add_global_arguments('-std=gnu++20', language: 'cpp')
elif cpp_compiler.get_id() == 'msvc'
    add_global_arguments('/W2', language: 'cpp')
endif

cpp_dir = 'src/'

cpp_sources = [
    cpp_dir + 'main.cc',
    cpp_dir + 'applicationInitFunctions.cc',
    cpp_dir + 'initUtilities.cc',
    
    cpp_dir + 'model.cc',
    cpp_dir + 'text_manager.cc',
    cpp_dir + 'camera.cpp'
]

sources = cpp_sources
vulkan_dep = dependency('vulkan')

glfw_dep = dependency('glfw3')
glm_dep = dependency('glm')

cmake = import('cmake')
opt_var = cmake.subproject_options()
opt_var.add_cmake_defines({
    'TINYGLTF_BUILD_LOADER_EXAMPLE': false,
    'TINYGLTF_HEADER_ONLY': false,
    'TINYGLTF_INSTALL': false
})

tinygltf = cmake.subproject('tinygltf', options: opt_var)
tinygltf_dep = tinygltf.dependency('tinygltf')

jolt_options = cmake.subproject_options()

jolt_options.add_cmake_defines({
    'TARGET_UNIT_TESTS': 'OFF',
    'TARGET_HELLO_WORLD': 'OFF',
    'TARGET_PERFORMANCE_TEST': 'OFF',
    'TARGET_SAMPLES': 'OFF',
    'TARGET_VIEWER': 'OFF',
    'USE_STATIC_MSVC_RUNTIME_LIBRARY': 'OFF',
    # 'USE_DOUBLE_PRECISION': 'ON'
})

jolt_subproj = cmake.subproject('jolt', options: jolt_options)

jolt_dep = jolt_subproj.dependency('Jolt')

include_dir = include_directories('Include')

executable(
    'main',
    sources: sources,
    include_directories: include_dir,
    dependencies: [vulkan_dep, glfw_dep, glm_dep, tinygltf_dep, jolt_dep],
)

slangc = find_program('slangc', required: true)

custom_target(
    'compile_slang_shader',
    input: 'res/shader.slang',
    output: 'shader.spv',
    command: [
        slangc,
        '@INPUT@',
        '-target', 'spirv',
        '-profile', 'spirv_1_4',
        '-emit-spirv-directly',
        '-fvk-use-entrypoint-name',
        '-entry', 'vertMain',
        '-entry', 'fragMain',
        '-o', '@OUTPUT@',
    ],
    build_by_default: true,
)

custom_target(
    'compile_slang_composition',
    input: 'res/composition.slang',
    output: 'composition.spv',
    command: [
        slangc,
        '@INPUT@',
        '-target', 'spirv',
        '-profile', 'spirv_1_4',
        '-emit-spirv-directly',
        '-fvk-use-entrypoint-name',
        '-entry', 'main',
        '-o', '@OUTPUT@',
    ],
    build_by_default: true,
)