#if defined(__INTELLISENSE__) || !defined(USE_CPP20_MODULES)
#define VULKAN_HPP_NO_STRUCT_CONSTRUCTORS
#include <vulkan/vulkan_raii.hpp>
#else
#define VULKAN_HPP_NO_STRUCT_CONSTRUCTORS
import vulkan_hpp;
#endif 

struct SkinComputeResources {
    // The mesh's original, unchanging rest-pose vertices
    vk::raii::Buffer input_vertex_buffer = nullptr;
    vk::raii::DeviceMemory input_vertex_memory = nullptr;
    uint32_t vertex_count;

    // Updated each frame with the current joint matrices
    vk::raii::Buffer joint_matrix_buffer = nullptr;
    vk::raii::DeviceMemory joint_matrix_memory = nullptr;
    uint32_t joint_count;

    // Written by compute, read by rasterizer/raytracer/physics
    vk::raii::Buffer output_vertex_buffer = nullptr;
    vk::raii::DeviceMemory output_vertex_memory = nullptr;

    // Descriptor set referencing all three buffers
    vk::raii::DescriptorSet descriptor_set = nullptr;
};

struct SkinPushConstants {
    unsigned int vertex_count;
};

class BonesMgr {
public:
    BonesMgr(vk::raii::Device &device, vk::raii::PhysicalDevice &physicalDevice);
    void dispatchSkinning(const vk::raii::CommandBuffer &cmd, const SkinComputeResources &skin);
    void insertSkinningBarrier(const vk::raii::CommandBuffer &cmd, vk::Buffer outputVertexBuffer);
private:
    vk::raii::Device* device;
    vk::raii::PhysicalDevice* physicalDevice;

    vk::raii::Pipeline computePipeline = nullptr;
    vk::raii::PipelineLayout pipelineLayout = nullptr;
    vk::raii::DescriptorSetLayout descriptorSetLayout = nullptr;
};