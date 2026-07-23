#include "bones.hh"
#include "vulkan/vulkan_raii.hpp"
#include "vulkan_utils.hpp"
#include <array>
#include <cstddef>

BonesMgr::BonesMgr(vk::raii::Device &device, vk::raii::PhysicalDevice &physicalDevice) {
    // Load the compiled .spv file
    auto shaderCode = VulkanUtils::readFile("shaders/skinning.spv");
    vk::raii::ShaderModule shaderModule = VulkanUtils::createShaderModule(shaderCode, device);

    // 1. Create Descriptor Set Layout
    // Must match the five bindings declared in skinning.slang.
    std::array<vk::DescriptorSetLayoutBinding, 5> bindings = {
        vk::DescriptorSetLayoutBinding{
            .binding = 0,
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eCompute},
        vk::DescriptorSetLayoutBinding{
            .binding = 1,
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eCompute},
        vk::DescriptorSetLayoutBinding{
            .binding = 2,
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eCompute},
        vk::DescriptorSetLayoutBinding{
            .binding = 3,
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eCompute},
        vk::DescriptorSetLayoutBinding{
            .binding = 4,
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eCompute}};

    vk::DescriptorSetLayoutCreateInfo layoutInfo{.bindingCount = static_cast<uint32_t>(bindings.size()), .pBindings = bindings.data()};
    descriptorSetLayout = vk::raii::DescriptorSetLayout(device, layoutInfo);

    // 2. Create Pipeline Layout (including push constants)
    vk::PushConstantRange pushRange{};
    pushRange.offset = 0;
    pushRange.size = sizeof(SkinPushConstants);

    std::array<vk::DescriptorSetLayout, 1> layouts = {
        *descriptorSetLayout,                    // Set 0
    };

    vk::PipelineLayoutCreateInfo pipelineLayoutInfo{
        .setLayoutCount = 1,
        .pSetLayouts = layouts.data(),
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &pushRange};

    pipelineLayout = vk::raii::PipelineLayout(device, pipelineLayoutInfo);

    // 3. Create Compute Pipeline
    vk::ComputePipelineCreateInfo pipelineInfo{
        .stage = vk::PipelineShaderStageCreateInfo{
            .stage = vk::ShaderStageFlagBits::eCompute,
            .module = *shaderModule,
            .pName = "main"},
        .layout = *pipelineLayout};

    computePipeline = vk::raii::Pipeline(device, nullptr, pipelineInfo);
}

void BonesMgr::dispatchSkinning(const vk::raii::CommandBuffer &cmd, const SkinComputeResources &skin) {
    // Bind the compute pipeline
    cmd.bindPipeline(vk::PipelineBindPoint::eCompute, *computePipeline);

    // Bind descriptor set
    cmd.bindDescriptorSets(
        vk::PipelineBindPoint::eCompute,
        *pipelineLayout,
        0,
        *skin.descriptor_set,
        nullptr
    );

    // Upload the vertex count as a push constant
    SkinPushConstants constants{
        .vertex_count = skin.vertex_count
    };
    cmd.pushConstants<SkinPushConstants>(
        *pipelineLayout,
        vk::ShaderStageFlagBits::eCompute,
        0,
        constants
    );

    // Calculate thread groups (64 vertices per workgroup)
    uint32_t group_count = (skin.vertex_count + 63) / 64;
    cmd.dispatch(group_count, 1, 1);
}

void BonesMgr::insertSkinningBarrier(const vk::raii::CommandBuffer &cmd, vk::Buffer outputVertexBuffer) {
    // Barrière mémoire pour le buffer de sommets de sortie
    vk::BufferMemoryBarrier2 barrier{
        .srcStageMask        = vk::PipelineStageFlagBits2::eComputeShader,
        .srcAccessMask       = vk::AccessFlagBits2::eShaderWrite,
        .dstStageMask        = vk::PipelineStageFlagBits2::eVertexInput | vk::PipelineStageFlagBits2::eRayTracingShaderKHR,
        .dstAccessMask       = vk::AccessFlagBits2::eVertexAttributeRead | vk::AccessFlagBits2::eShaderRead,
        .srcQueueFamilyIndex = vk::QueueFamilyIgnored,
        .dstQueueFamilyIndex = vk::QueueFamilyIgnored,
        .buffer              = outputVertexBuffer,
        .offset              = 0,
        .size                = vk::WholeSize
    };

    vk::DependencyInfo depInfo{
        .bufferMemoryBarrierCount = 1,
        .pBufferMemoryBarriers    = &barrier
    };

    cmd.pipelineBarrier2(depInfo);
}