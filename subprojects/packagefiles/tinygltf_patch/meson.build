project('tinygltf', 'cpp', version: '2.9.6', default_options: ['cpp_std=c++20'])

tinygltf_inc = include_directories('.')

tinygltf_lib = static_library('tinygltf',
    'tiny_gltf.cc',
    include_directories: tinygltf_inc
)

tinygltf_dep = declare_dependency(
    include_directories: tinygltf_inc,
    link_with: tinygltf_lib
)
