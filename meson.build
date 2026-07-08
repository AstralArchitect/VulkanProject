project(
    'Vulkan-Project',
    'c',
    'cpp',
    version: '0.1.0',
    default_options: ['cpp_std=c++20', 'default_library=static'],
)

cpp_compiler = meson.get_compiler('cpp')

# Gestion des arguments selon le compilateur
if cpp_compiler.get_id() == 'gcc'
    add_global_arguments('-std=gnu++20', language: 'cpp')
elif cpp_compiler.get_id() == 'msvc'
    add_global_arguments('/W2', language: 'cpp')
endif

# --- Chemins des sources ---
cpp_dir = 'src/'
tools_dir = cpp_dir + 'tools/'

cpp_sources = [
    cpp_dir + 'main.cc',
    cpp_dir + 'applicationInitFunctions.cc',
    cpp_dir + 'initUtilities.cc',
]

sources = cpp_sources

# --- Gestion des Dépendances ---

# 1. Vulkan : Meson le trouve automatiquement via la variable VULKAN_SDK
vulkan_dep = dependency('vulkan')

# 2. GLFW & GLM : Puisque tu gères les fichiers manuellement, 
# Meson les trouvera s'ils sont dans ton PATH ou si tu utilises des Wraps.
glfw_dep = dependency('glfw3')
glm_dep = dependency('glm')

# --- Inclusion des dossiers de headers ---
include_dir = include_directories('Include')

# --- Exécutable final ---
executable(
    'main',
    sources: sources,
    include_directories: include_dir,
    dependencies: [vulkan_dep, glfw_dep, glm_dep],
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
        '-o', '@OUTPUT@' # Le fichier de destination généré par Meson
,
    ],
    build_by_default: true # Force la compilation même si l'exécutable C++ n'en dépend pas directement
,
)