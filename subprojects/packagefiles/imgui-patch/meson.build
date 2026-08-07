project('imgui', 'cpp', version: '1.90.4', default_options: ['cpp_std=c++20'])

glfw_dep = dependency('glfw3')
vulkan_dep = dependency('vulkan')

imgui_inc = include_directories('.', 'backends')

imgui_sources = files(
    'imgui.cpp',
    'imgui_draw.cpp',
    'imgui_tables.cpp',
    'imgui_widgets.cpp',
    'backends/imgui_impl_glfw.cpp',
    'backends/imgui_impl_vulkan.cpp',
)

imgui_dep = declare_dependency(
    include_directories: imgui_inc,
    sources: imgui_sources,
    dependencies: [glfw_dep, vulkan_dep]
)
