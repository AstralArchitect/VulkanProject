project(
    'Vulkan-Project',
    'c',
    'cpp',
    version: '1.0.0',
    default_options: ['cpp_std=c++20', 'default_library=static'],
)

cpp_compiler = meson.get_compiler('cpp')

# Arguments globaux du projet (indispensables pour Vulkan/GLM)
add_project_arguments('-DGLM_FORCE_DEFAULT_ALIGNED_GENTYPES', '-DGLM_FORCE_DEPTH_ZERO_TO_ONE', language: 'cpp')

# Gestion des arguments selon le compilateur
if cpp_compiler.get_id() == 'gcc'
    add_global_arguments('-std=gnu++20', language: 'cpp')
elif cpp_compiler.get_id() == 'msvc'
    add_global_arguments('/W2', language: 'cpp')
endif

# --- Chemins des sources ---
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

# --- Gestion des Dépendances ---

# 1. Vulkan : Meson le trouve automatiquement via la variable VULKAN_SDK
vulkan_dep = dependency('vulkan')

# 2. GLFW & GLM : Puisque tu gères les fichiers manuellement, 
# Meson les trouvera s'ils sont dans ton PATH ou si tu utilises des Wraps.
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

# --- Inclusion des dossiers de headers ---
include_dir = include_directories('Include')

# --- Exécutable final ---
executable(
    'main',
    sources: sources,
    include_directories: include_dir,
    dependencies: [vulkan_dep, glfw_dep, glm_dep, tinygltf_dep],
)

# --- Compiler les shaders ---
# 1. Trouver l'exécutable slangc sur le système
slangc = find_program('slangc', required: true)

# 2. Créer une cible personnalisée pour compiler le shader
custom_target(
    'compile_slang_shader',
    input: 'res/shader.slang',
    output: 'shader.spv',
    command: [
        slangc,
        '@INPUT@', # Le fichier source (res/shader.slang)
        '-target', 'spirv', # On spécifie qu'on veut du SPIR-V pour Vulkan
        '-profile', 'spirv_1_4',
        '-emit-spirv-directly',
        '-fvk-use-entrypoint-name',
        '-entry', 'vertMain',
        '-entry', 'fragMain',
        '-o', '@OUTPUT@', # Le fichier de destination généré par Meson
    ],
    build_by_default: true, # Force la compilation même si l'exécutable C++ n'en dépend pas directement
)

custom_target(
    'compile_slang_composition',
    input: 'res/composition.slang',
    output: 'composition.spv',
    command: [
        slangc,
        '@INPUT@', # Le fichier source (res/shader.slang)
        '-target', 'spirv', # On spécifie qu'on veut du SPIR-V pour Vulkan
        '-profile', 'spirv_1_4',
        '-emit-spirv-directly',
        '-fvk-use-entrypoint-name',
        '-entry', 'main',
        '-o', '@OUTPUT@', # Le fichier de destination généré par Meson
    ],
    build_by_default: true, # Force la compilation même si l'exécutable C++ n'en dépend pas directement
)