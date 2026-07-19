#include "ffx-mgr.hh"

#include "vulkan_utils.hpp"

#include <cstdint>
#include <cstring>

extern const int MAX_FRAMES_IN_FLIGHT;

FFXMgr::FFXMgr(vk::raii::Device &device, vk::raii::PhysicalDevice &physicalDevice, vk::raii::CommandPool &commandPool, uint32_t width, uint32_t height) 
    : device(&device), physicalDevice(&physicalDevice), commandPool(&commandPool), m_width(width), m_height(height) 
{
    vk::ImageUsageFlags ffxUsage = vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst;
    vk::MemoryPropertyFlags ffxMemProps = vk::MemoryPropertyFlagBits::eDeviceLocal;

    // Helper lambda to easily stamp out full resolution buffers
    auto createFullResImage = [&](vk::Format format, vk::raii::Image& img, vk::raii::DeviceMemory& mem, vk::raii::ImageView& view, vk::ImageAspectFlags aspect = vk::ImageAspectFlagBits::eColor, vk::ImageUsageFlags extraUsage = {}, vk::ImageUsageFlags removeUsage = {}) {
        vk::ImageUsageFlags finalUsage = (ffxUsage | extraUsage) & ~removeUsage;
        auto [image, memory] = VulkanUtils::createImage(device, physicalDevice, m_width, m_height, format, vk::ImageTiling::eOptimal, finalUsage, ffxMemProps, vk::SampleCountFlagBits::e1);
        img = std::move(image);
        mem = std::move(memory);
        view = VulkanUtils::createImageView(device, img, format, aspect);
    };

    // 1. Intermediate Buffers
    createFullResImage(vk::Format::eR16G16B16A16Sfloat, images.ffxReprojectedRadianceImage, images.ffxReprojectedRadianceMemory, images.ffxReprojectedRadianceView);
    createFullResImage(vk::Format::eR16Sfloat,          images.ffxReprojectedVarianceImage, images.ffxReprojectedVarianceMemory, images.ffxReprojectedVarianceView);
    createFullResImage(vk::Format::eR16G16B16A16Sfloat, images.ffxPrefilteredRadianceImage, images.ffxPrefilteredRadianceMemory, images.ffxPrefilteredRadianceView);
    createFullResImage(vk::Format::eR16Sfloat,          images.ffxPrefilteredVarianceImage, images.ffxPrefilteredVarianceMemory, images.ffxPrefilteredVarianceView);
    createFullResImage(vk::Format::eR16Sfloat,          images.ffxFinalVarianceImage,       images.ffxFinalVarianceMemory,       images.ffxFinalVarianceView);

    // 2. Average Radiance (1/8th resolution calculation)
    uint32_t avgWidth = (m_width + 7) / 8;
    uint32_t avgHeight = (m_height + 7) / 8;
    auto [avgImg, avgMem] = VulkanUtils::createImage(device, physicalDevice, avgWidth, avgHeight, vk::Format::eR16G16B16A16Sfloat, vk::ImageTiling::eOptimal, ffxUsage, ffxMemProps, vk::SampleCountFlagBits::e1);
    images.ffxAverageRadianceImage = std::move(avgImg);
    images.ffxAverageRadianceMemory = std::move(avgMem);
    images.ffxAverageRadianceView = VulkanUtils::createImageView(device, images.ffxAverageRadianceImage, vk::Format::eR16G16B16A16Sfloat, vk::ImageAspectFlagBits::eColor);

    // 3. History Buffers (Ping-Pong arrays)
    for (int i = 0; i < 2; i++) {
        createFullResImage(vk::Format::eR16G16B16A16Sfloat, images.ffxHistoryRadianceImages[i],  images.ffxHistoryRadianceMemory[i],  images.ffxHistoryRadianceViews[i]);
        createFullResImage(vk::Format::eD32Sfloat,          images.ffxHistoryDepthImages[i],     images.ffxHistoryDepthMemory[i],     images.ffxHistoryDepthViews[i], vk::ImageAspectFlagBits::eDepth, {}, vk::ImageUsageFlagBits::eStorage);
        createFullResImage(vk::Format::eR16G16B16A16Sfloat, images.ffxHistoryNormalImages[i],    images.ffxHistoryNormalMemory[i],    images.ffxHistoryNormalViews[i]);
        createFullResImage(vk::Format::eR16G16B16A16Sfloat, images.ffxHistoryRoughnessImages[i], images.ffxHistoryRoughnessMemory[i], images.ffxHistoryRoughnessViews[i]);
        createFullResImage(vk::Format::eR16Sfloat,          images.ffxSampleCountImages[i],      images.ffxSampleCountMemory[i],      images.ffxSampleCountViews[i]);
    }

    createDescriptorPool();
    createReprojShaderResources();
    createPrefilShaderResources();
    createResolvShaderResources();

    vk::DeviceSize bufferSize = sizeof(FFXConstants);

    std::tie(constantsBuffer, constantsBufferMemory) = VulkanUtils::createBuffer(
    device, physicalDevice, bufferSize,
    vk::BufferUsageFlagBits::eShaderDeviceAddress | vk::BufferUsageFlagBits::eUniformBuffer,
    vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);

    vk::SamplerCreateInfo samplerInfo{
        .magFilter     = vk::Filter::eLinear,
        .minFilter     = vk::Filter::eLinear,
        .mipmapMode    = vk::SamplerMipmapMode::eLinear,
        .addressModeU  = vk::SamplerAddressMode::eClampToEdge,
        .addressModeV  = vk::SamplerAddressMode::eClampToEdge,
        .addressModeW  = vk::SamplerAddressMode::eClampToEdge,
        .mipLodBias    = 0.0f,
        .maxAnisotropy = 1.0f,
        .minLod        = 0.0f,
        .maxLod        = 1.0f
    };

    linearSampler = vk::raii::Sampler(device, samplerInfo);
}

void FFXMgr::createDescriptorPool() {
    descriptorPool = nullptr;

    std::array<vk::DescriptorPoolSize, 4> poolSizes = {
        //                                                          R    P   RT  (reproject, prefilter, resolve temporal)
        vk::DescriptorPoolSize{vk::DescriptorType::eSampledImage,   (12 + 6 + 6) * 2},
        vk::DescriptorPoolSize{vk::DescriptorType::eStorageImage,   (4  + 2 + 2) * 2},
        vk::DescriptorPoolSize{vk::DescriptorType::eSampler,        (1  + 1 + 1) * 2},
        vk::DescriptorPoolSize{vk::DescriptorType::eUniformBuffer,  (1  + 1 + 1) * 2}
    };
    vk::DescriptorPoolCreateInfo poolInfo{
        .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
        .maxSets = 6, // 2 pour Reproject + 2 pour Prefilter + 2 pour Resolve
        .poolSizeCount = static_cast<uint32_t>(poolSizes.size()),
        .pPoolSizes = poolSizes.data()};
        
    descriptorPool = vk::raii::DescriptorPool(*device, poolInfo);
}

void FFXMgr::createReprojShaderResources()
{
    descriptorSets[0].clear();
    pipelines[0] = nullptr;
    pipelineLayouts[0] = nullptr;
    descriptorSetLayouts[0] = nullptr;

    std::array<vk::DescriptorSetLayoutBinding, 18> bindings;

    // Bindings 0-11 are VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE
    for (int i = 0; i < 12; i++) {
        bindings[i] = vk::DescriptorSetLayoutBinding{
            .binding = (uint32_t)i,
            .descriptorType = vk::DescriptorType::eSampledImage,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eCompute};
    }
    // Bindings 12-15 are VK_DESCRIPTOR_TYPE_STORAGE_IMAGE
    for (int i = 12; i < 16; i++) {
        bindings[i] = vk::DescriptorSetLayoutBinding{
            .binding = (uint32_t)i,
            .descriptorType = vk::DescriptorType::eStorageImage,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eCompute};
    }
    // Bindings 16 and 17 are VK_DESCRIPTOR_TYPE_SAMPLER and VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
    bindings[16] = vk::DescriptorSetLayoutBinding{
            .binding = 16,
            .descriptorType = vk::DescriptorType::eSampler,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eCompute};
    bindings[17] = vk::DescriptorSetLayoutBinding{
            .binding = 17,
            .descriptorType = vk::DescriptorType::eUniformBuffer,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eCompute};
 
    vk::DescriptorSetLayoutCreateInfo layoutInfo{
        .bindingCount = static_cast<uint32_t>(bindings.size()),
        .pBindings = bindings.data()};
    descriptorSetLayouts[0] = vk::raii::DescriptorSetLayout(*device, layoutInfo);

    vk::PipelineLayoutCreateInfo pipelineLayoutInfo{
        .setLayoutCount = 1,
        .pSetLayouts = &(*descriptorSetLayouts[0])};
    pipelineLayouts[0] = vk::raii::PipelineLayout(*device, pipelineLayoutInfo);

    vk::raii::ShaderModule shaderModule = VulkanUtils::createShaderModule(VulkanUtils::readFile("builddir/reprojection.spv"), *device);
    vk::ComputePipelineCreateInfo pipelineInfo{
        .stage = vk::PipelineShaderStageCreateInfo{
            .stage = vk::ShaderStageFlagBits::eCompute,
            .module = *shaderModule,
            .pName = "main"},
        .layout = *pipelineLayouts[0]};
    pipelines[0] = vk::raii::Pipeline(*device, nullptr, pipelineInfo);

    uint32_t setCount = 2;

    std::vector<vk::DescriptorSetLayout> layouts(setCount, *descriptorSetLayouts[0]);
    vk::DescriptorSetAllocateInfo allocInfo{
        .descriptorPool = *descriptorPool,
        .descriptorSetCount = setCount,
        .pSetLayouts = layouts.data()};
    descriptorSets[0] = vk::raii::DescriptorSets(*device, allocInfo);
}

void FFXMgr::updateReprojDescriptorSets(
    vk::ImageView inputRadiance,
    vk::ImageView inputDepth,
    vk::ImageView inputNormal,
    vk::ImageView inputRoughness,
    vk::ImageView inputMotionVectors,
    vk::ImageView inputRayLength)
{
    // Update both descriptor sets (Ping-Pong 0 and 1)
    for (int i = 0; i < 2; i++) {
        vk::DescriptorImageInfo inputRadianceInfo{
            .sampler = nullptr,
            .imageView = inputRadiance,
            .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal};
            
        vk::DescriptorImageInfo inputDepthInfo{
            .sampler = nullptr,
            .imageView = inputDepth,
            .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal};
            
        vk::DescriptorImageInfo inputNormalInfo{
            .sampler = nullptr,
            .imageView = inputNormal,
            .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal};
            
        vk::DescriptorImageInfo inputRoughnessInfo{
            .sampler = nullptr,
            .imageView = inputRoughness,
            .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal};
            
        vk::DescriptorImageInfo inputMotionVectorsInfo{
            .sampler = nullptr,
            .imageView = inputMotionVectors,
            .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal};
            
        vk::DescriptorImageInfo inputRayLengthInfo{
            .sampler = nullptr,
            .imageView = inputRayLength,
            .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal};

        // History: read from History[i]
        vk::DescriptorImageInfo historyRadianceInfo{
            .sampler = nullptr,
            .imageView = *images.ffxHistoryRadianceViews[i],
            .imageLayout = vk::ImageLayout::eGeneral};
            
        vk::DescriptorImageInfo historyDepthInfo{
            .sampler = nullptr,
            .imageView = *images.ffxHistoryDepthViews[i],
            .imageLayout = vk::ImageLayout::eGeneral};
            
        vk::DescriptorImageInfo historyNormalInfo{
            .sampler = nullptr,
            .imageView = *images.ffxHistoryNormalViews[i],
            .imageLayout = vk::ImageLayout::eGeneral};
            
        vk::DescriptorImageInfo historyRoughnessInfo{
            .sampler = nullptr,
            .imageView = *images.ffxHistoryRoughnessViews[i],
            .imageLayout = vk::ImageLayout::eGeneral};

        vk::DescriptorImageInfo historyVarianceInfo{
            .sampler = nullptr,
            .imageView = *images.ffxFinalVarianceView,
            .imageLayout = vk::ImageLayout::eGeneral};
            
        vk::DescriptorImageInfo historyNumSamplesInfo{
            .sampler = nullptr,
            .imageView = *images.ffxSampleCountViews[i],
            .imageLayout = vk::ImageLayout::eGeneral};

        // Outputs (Intermediate)
        vk::DescriptorImageInfo outReprojectedRadianceInfo{
            .sampler = nullptr,
            .imageView = *images.ffxReprojectedRadianceView,
            .imageLayout = vk::ImageLayout::eGeneral};
            
        vk::DescriptorImageInfo outAverageRadianceInfo{
            .sampler = nullptr,
            .imageView = *images.ffxAverageRadianceView,
            .imageLayout = vk::ImageLayout::eGeneral};
            
        vk::DescriptorImageInfo outVarianceInfo{
            .sampler = nullptr,
            .imageView = *images.ffxReprojectedVarianceView,
            .imageLayout = vk::ImageLayout::eGeneral};
            
        uint32_t writeIdx = (i + 1) % 2;
        vk::DescriptorImageInfo outNumSamplesInfo{
            .sampler = nullptr,
            .imageView = *images.ffxSampleCountViews[writeIdx],
            .imageLayout = vk::ImageLayout::eGeneral};

        vk::DescriptorImageInfo samplerInfo{
            .sampler = linearSampler,
            .imageView = nullptr,
            .imageLayout = vk::ImageLayout::eUndefined};

        vk::DescriptorBufferInfo bufferInfo{
            .buffer = constantsBuffer,
            .offset = 0,
            .range = VK_WHOLE_SIZE};

        std::array<vk::WriteDescriptorSet, 18> descriptorWrites = {
            vk::WriteDescriptorSet{.dstSet = *descriptorSets[0][i], .dstBinding = 0, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eSampledImage, .pImageInfo = &inputRadianceInfo},
            vk::WriteDescriptorSet{.dstSet = *descriptorSets[0][i], .dstBinding = 1, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eSampledImage, .pImageInfo = &inputDepthInfo},
            vk::WriteDescriptorSet{.dstSet = *descriptorSets[0][i], .dstBinding = 2, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eSampledImage, .pImageInfo = &inputNormalInfo},
            vk::WriteDescriptorSet{.dstSet = *descriptorSets[0][i], .dstBinding = 3, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eSampledImage, .pImageInfo = &inputRoughnessInfo},
            vk::WriteDescriptorSet{.dstSet = *descriptorSets[0][i], .dstBinding = 4, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eSampledImage, .pImageInfo = &inputMotionVectorsInfo},
            vk::WriteDescriptorSet{.dstSet = *descriptorSets[0][i], .dstBinding = 5, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eSampledImage, .pImageInfo = &inputRayLengthInfo},
            vk::WriteDescriptorSet{.dstSet = *descriptorSets[0][i], .dstBinding = 6, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eSampledImage, .pImageInfo = &historyRadianceInfo},
            vk::WriteDescriptorSet{.dstSet = *descriptorSets[0][i], .dstBinding = 7, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eSampledImage, .pImageInfo = &historyDepthInfo},
            vk::WriteDescriptorSet{.dstSet = *descriptorSets[0][i], .dstBinding = 8, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eSampledImage, .pImageInfo = &historyNormalInfo},
            vk::WriteDescriptorSet{.dstSet = *descriptorSets[0][i], .dstBinding = 9, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eSampledImage, .pImageInfo = &historyRoughnessInfo},
            vk::WriteDescriptorSet{.dstSet = *descriptorSets[0][i], .dstBinding = 10, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eSampledImage, .pImageInfo = &historyVarianceInfo},
            vk::WriteDescriptorSet{.dstSet = *descriptorSets[0][i], .dstBinding = 11, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eSampledImage, .pImageInfo = &historyNumSamplesInfo},
            vk::WriteDescriptorSet{.dstSet = *descriptorSets[0][i], .dstBinding = 12, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eStorageImage, .pImageInfo = &outReprojectedRadianceInfo},
            vk::WriteDescriptorSet{.dstSet = *descriptorSets[0][i], .dstBinding = 13, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eStorageImage, .pImageInfo = &outAverageRadianceInfo},
            vk::WriteDescriptorSet{.dstSet = *descriptorSets[0][i], .dstBinding = 14, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eStorageImage, .pImageInfo = &outVarianceInfo},
            vk::WriteDescriptorSet{.dstSet = *descriptorSets[0][i], .dstBinding = 15, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eStorageImage, .pImageInfo = &outNumSamplesInfo},
            vk::WriteDescriptorSet{.dstSet = *descriptorSets[0][i], .dstBinding = 16, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eSampler, .pImageInfo = &samplerInfo},
            vk::WriteDescriptorSet{.dstSet = *descriptorSets[0][i], .dstBinding = 17, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eUniformBuffer, .pBufferInfo = &bufferInfo}
        };

        device->updateDescriptorSets(descriptorWrites, nullptr);
    }
}

void FFXMgr::createPrefilShaderResources()
{
    descriptorSets[1].clear();
    pipelines[1] = nullptr;
    pipelineLayouts[1] = nullptr;
    descriptorSetLayouts[1] = nullptr;

    std::array<vk::DescriptorSetLayoutBinding, 10> bindings;

    // Bindings 0-5 are VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE
    for (int i = 0; i < 6; i++) {
        bindings[i] = vk::DescriptorSetLayoutBinding{
            .binding = (uint32_t)i,
            .descriptorType = vk::DescriptorType::eSampledImage,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eCompute};
    }
    // Bindings 6-7 are VK_DESCRIPTOR_TYPE_STORAGE_IMAGE
    for (int i = 6; i < 8; i++) {
        bindings[i] = vk::DescriptorSetLayoutBinding{
            .binding = (uint32_t)i,
            .descriptorType = vk::DescriptorType::eStorageImage,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eCompute};
    }
    // Binding 8 is VK_DESCRIPTOR_TYPE_SAMPLER
    bindings[8] = vk::DescriptorSetLayoutBinding{
            .binding = 8,
            .descriptorType = vk::DescriptorType::eSampler,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eCompute};
    // Binding 9 is VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
    bindings[9] = vk::DescriptorSetLayoutBinding{
            .binding = 9,
            .descriptorType = vk::DescriptorType::eUniformBuffer,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eCompute};
 
    vk::DescriptorSetLayoutCreateInfo layoutInfo{
        .bindingCount = static_cast<uint32_t>(bindings.size()),
        .pBindings = bindings.data()};
    descriptorSetLayouts[1] = vk::raii::DescriptorSetLayout(*device, layoutInfo);

    vk::PipelineLayoutCreateInfo pipelineLayoutInfo{
        .setLayoutCount = 1,
        .pSetLayouts = &(*descriptorSetLayouts[1])};
    pipelineLayouts[1] = vk::raii::PipelineLayout(*device, pipelineLayoutInfo);

    vk::raii::ShaderModule shaderModule = VulkanUtils::createShaderModule(VulkanUtils::readFile("builddir/prefilter.spv"), *device);
    vk::ComputePipelineCreateInfo pipelineInfo{
        .stage = vk::PipelineShaderStageCreateInfo{
            .stage = vk::ShaderStageFlagBits::eCompute,
            .module = *shaderModule,
            .pName = "main"},
        .layout = *pipelineLayouts[1]};
    pipelines[1] = vk::raii::Pipeline(*device, nullptr, pipelineInfo);

    uint32_t setCount = 2;
    std::vector<vk::DescriptorSetLayout> layouts(setCount, *descriptorSetLayouts[1]);
    vk::DescriptorSetAllocateInfo allocInfo{
        .descriptorPool = *descriptorPool,
        .descriptorSetCount = setCount,
        .pSetLayouts = layouts.data()};
    descriptorSets[1] = vk::raii::DescriptorSets(*device, allocInfo);
}

void FFXMgr::createResolvShaderResources()
{
    descriptorSets[2].clear();
    pipelines[2] = nullptr;
    pipelineLayouts[2] = nullptr;
    descriptorSetLayouts[2] = nullptr;

    std::array<vk::DescriptorSetLayoutBinding, 10> bindings;

    // Bindings 0-5 are VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE
    for (int i = 0; i < 6; i++) {
        bindings[i] = vk::DescriptorSetLayoutBinding{
            .binding = (uint32_t)i,
            .descriptorType = vk::DescriptorType::eSampledImage,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eCompute};
    }
    // Bindings 6-7 are VK_DESCRIPTOR_TYPE_STORAGE_IMAGE
    for (int i = 6; i < 8; i++) {
        bindings[i] = vk::DescriptorSetLayoutBinding{
            .binding = (uint32_t)i,
            .descriptorType = vk::DescriptorType::eStorageImage,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eCompute};
    }
    // Bindings 8 and 9 are VK_DESCRIPTOR_TYPE_SAMPLER and VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
    bindings[8] = vk::DescriptorSetLayoutBinding{
            .binding = 8,
            .descriptorType = vk::DescriptorType::eSampler,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eCompute};
    bindings[9] = vk::DescriptorSetLayoutBinding{
            .binding = 9,
            .descriptorType = vk::DescriptorType::eUniformBuffer,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eCompute};
 
    vk::DescriptorSetLayoutCreateInfo layoutInfo{
        .bindingCount = static_cast<uint32_t>(bindings.size()),
        .pBindings = bindings.data()};
    descriptorSetLayouts[2] = vk::raii::DescriptorSetLayout(*device, layoutInfo);

    vk::PipelineLayoutCreateInfo pipelineLayoutInfo{
        .setLayoutCount = 1,
        .pSetLayouts = &(*descriptorSetLayouts[2])};
    pipelineLayouts[2] = vk::raii::PipelineLayout(*device, pipelineLayoutInfo);

    vk::raii::ShaderModule shaderModule = VulkanUtils::createShaderModule(VulkanUtils::readFile("builddir/resolve_temporal.spv"), *device);
    vk::ComputePipelineCreateInfo pipelineInfo{
        .stage = vk::PipelineShaderStageCreateInfo{
            .stage = vk::ShaderStageFlagBits::eCompute,
            .module = *shaderModule,
            .pName = "main"},
        .layout = *(pipelineLayouts[2])};
    pipelines[2] = vk::raii::Pipeline(*device, nullptr, pipelineInfo);

    uint32_t setCount = 2;
    std::vector<vk::DescriptorSetLayout> layouts(setCount, *descriptorSetLayouts[2]);
    vk::DescriptorSetAllocateInfo allocInfo{
        .descriptorPool = *descriptorPool,
        .descriptorSetCount = setCount,
        .pSetLayouts = layouts.data()};
    descriptorSets[2] = vk::raii::DescriptorSets(*device, allocInfo);
}

void FFXMgr::updatePrefilDescriptorSets(
    vk::ImageView inputRadiance,
    vk::ImageView inputDepth,
    vk::ImageView inputNormal,
    vk::ImageView inputRoughness)
{
    for (int i = 0; i < 2; i++) {
        vk::DescriptorImageInfo inRadianceInfo{.sampler = nullptr, .imageView = inputRadiance, .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal};
        vk::DescriptorImageInfo inVarianceInfo{.sampler = nullptr, .imageView = *images.ffxReprojectedVarianceView, .imageLayout = vk::ImageLayout::eGeneral};
        vk::DescriptorImageInfo inNormalInfo{.sampler = nullptr, .imageView = inputNormal, .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal};
        vk::DescriptorImageInfo inDepthInfo{.sampler = nullptr, .imageView = inputDepth, .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal};
        vk::DescriptorImageInfo inRoughnessInfo{.sampler = nullptr, .imageView = inputRoughness, .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal};
        vk::DescriptorImageInfo inAverageInfo{.sampler = nullptr, .imageView = *images.ffxAverageRadianceView, .imageLayout = vk::ImageLayout::eGeneral};
        
        vk::DescriptorImageInfo outRadianceInfo{.sampler = nullptr, .imageView = *images.ffxPrefilteredRadianceView, .imageLayout = vk::ImageLayout::eGeneral};
        vk::DescriptorImageInfo outVarianceInfo{.sampler = nullptr, .imageView = *images.ffxPrefilteredVarianceView, .imageLayout = vk::ImageLayout::eGeneral};

        vk::DescriptorImageInfo samplerInfo{.sampler = linearSampler, .imageView = nullptr, .imageLayout = vk::ImageLayout::eUndefined};
        vk::DescriptorBufferInfo bufferInfo{.buffer = constantsBuffer, .offset = 0, .range = VK_WHOLE_SIZE};

        std::array<vk::WriteDescriptorSet, 10> descriptorWrites = {
            vk::WriteDescriptorSet{.dstSet = *descriptorSets[1][i], .dstBinding = 0, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eSampledImage, .pImageInfo = &inRadianceInfo},
            vk::WriteDescriptorSet{.dstSet = *descriptorSets[1][i], .dstBinding = 1, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eSampledImage, .pImageInfo = &inVarianceInfo},
            vk::WriteDescriptorSet{.dstSet = *descriptorSets[1][i], .dstBinding = 2, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eSampledImage, .pImageInfo = &inNormalInfo},
            vk::WriteDescriptorSet{.dstSet = *descriptorSets[1][i], .dstBinding = 3, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eSampledImage, .pImageInfo = &inDepthInfo},
            vk::WriteDescriptorSet{.dstSet = *descriptorSets[1][i], .dstBinding = 4, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eSampledImage, .pImageInfo = &inRoughnessInfo},
            vk::WriteDescriptorSet{.dstSet = *descriptorSets[1][i], .dstBinding = 5, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eSampledImage, .pImageInfo = &inAverageInfo},
            vk::WriteDescriptorSet{.dstSet = *descriptorSets[1][i], .dstBinding = 6, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eStorageImage, .pImageInfo = &outRadianceInfo},
            vk::WriteDescriptorSet{.dstSet = *descriptorSets[1][i], .dstBinding = 7, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eStorageImage, .pImageInfo = &outVarianceInfo},
            vk::WriteDescriptorSet{.dstSet = *descriptorSets[1][i], .dstBinding = 8, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eSampler, .pImageInfo = &samplerInfo},
            vk::WriteDescriptorSet{.dstSet = *descriptorSets[1][i], .dstBinding = 9, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eUniformBuffer, .pBufferInfo = &bufferInfo}
        };

        device->updateDescriptorSets(descriptorWrites, nullptr);
    }
}

void FFXMgr::updateResolvDescriptorSets(vk::ImageView inputRoughness)
{
    for (int i = 0; i < 2; i++) {
        vk::DescriptorImageInfo inRadianceInfo{.sampler = nullptr, .imageView = *images.ffxPrefilteredRadianceView, .imageLayout = vk::ImageLayout::eGeneral};
        vk::DescriptorImageInfo inVarianceInfo{.sampler = nullptr, .imageView = *images.ffxPrefilteredVarianceView, .imageLayout = vk::ImageLayout::eGeneral};
        vk::DescriptorImageInfo inRoughnessInfo{.sampler = nullptr, .imageView = inputRoughness, .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal};
        vk::DescriptorImageInfo inReprojectedInfo{.sampler = nullptr, .imageView = *images.ffxReprojectedRadianceView, .imageLayout = vk::ImageLayout::eGeneral};
        vk::DescriptorImageInfo inAverageInfo{.sampler = nullptr, .imageView = *images.ffxAverageRadianceView, .imageLayout = vk::ImageLayout::eGeneral};
        uint32_t readIdx = (i + 1) % 2;
        vk::DescriptorImageInfo inNumSamplesInfo{.sampler = nullptr, .imageView = *images.ffxSampleCountViews[readIdx], .imageLayout = vk::ImageLayout::eGeneral};
        
        // Output writes to the NEXT frame's history
        uint32_t writeIdx = (i + 1) % 2;
        vk::DescriptorImageInfo outRadianceInfo{.sampler = nullptr, .imageView = *images.ffxHistoryRadianceViews[writeIdx], .imageLayout = vk::ImageLayout::eGeneral};
        vk::DescriptorImageInfo outVarianceInfo{.sampler = nullptr, .imageView = *images.ffxFinalVarianceView, .imageLayout = vk::ImageLayout::eGeneral};

        vk::DescriptorImageInfo samplerInfo{.sampler = linearSampler, .imageView = nullptr, .imageLayout = vk::ImageLayout::eUndefined};
        vk::DescriptorBufferInfo bufferInfo{.buffer = constantsBuffer, .offset = 0, .range = VK_WHOLE_SIZE};

        std::array<vk::WriteDescriptorSet, 10> descriptorWrites = {
            vk::WriteDescriptorSet{.dstSet = *descriptorSets[2][i], .dstBinding = 0, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eSampledImage, .pImageInfo = &inRadianceInfo},
            vk::WriteDescriptorSet{.dstSet = *descriptorSets[2][i], .dstBinding = 1, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eSampledImage, .pImageInfo = &inVarianceInfo},
            vk::WriteDescriptorSet{.dstSet = *descriptorSets[2][i], .dstBinding = 2, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eSampledImage, .pImageInfo = &inRoughnessInfo},
            vk::WriteDescriptorSet{.dstSet = *descriptorSets[2][i], .dstBinding = 3, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eSampledImage, .pImageInfo = &inReprojectedInfo},
            vk::WriteDescriptorSet{.dstSet = *descriptorSets[2][i], .dstBinding = 4, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eSampledImage, .pImageInfo = &inAverageInfo},
            vk::WriteDescriptorSet{.dstSet = *descriptorSets[2][i], .dstBinding = 5, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eSampledImage, .pImageInfo = &inNumSamplesInfo},
            vk::WriteDescriptorSet{.dstSet = *descriptorSets[2][i], .dstBinding = 6, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eStorageImage, .pImageInfo = &outRadianceInfo},
            vk::WriteDescriptorSet{.dstSet = *descriptorSets[2][i], .dstBinding = 7, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eStorageImage, .pImageInfo = &outVarianceInfo},
            vk::WriteDescriptorSet{.dstSet = *descriptorSets[2][i], .dstBinding = 8, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eSampler, .pImageInfo = &samplerInfo},
            vk::WriteDescriptorSet{.dstSet = *descriptorSets[2][i], .dstBinding = 9, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eUniformBuffer, .pBufferInfo = &bufferInfo}
        };

        device->updateDescriptorSets(descriptorWrites, nullptr);
    }
}

void FFXMgr::dispatchDenoiser(vk::raii::CommandBuffer& cmd, uint32_t frameIndex, uint32_t width, uint32_t height, vk::Image srcDepth, vk::Image srcNormal, vk::Image srcRoughness)
{
    uint32_t pingPongIdx = frameIndex % 2;
    uint32_t nextPingPongIdx = (frameIndex + 1) % 2;
    uint32_t groupCountX = (width + 7) / 8;
    uint32_t groupCountY = (height + 7) / 8;

    if (!m_initialLayoutTransitionDone) {
        auto transitionToGeneral = [&](vk::raii::Image& img, vk::ImageAspectFlags aspect = vk::ImageAspectFlagBits::eColor) {
            if (!*img) return;
            vk::ImageMemoryBarrier barrier{
                .srcAccessMask = {},
                .dstAccessMask = vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite,
                .oldLayout = vk::ImageLayout::eUndefined,
                .newLayout = vk::ImageLayout::eGeneral,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = *img,
                .subresourceRange = {
                    .aspectMask = aspect,
                    .baseMipLevel = 0,
                    .levelCount = 1,
                    .baseArrayLayer = 0,
                    .layerCount = 1
                }
            };
            cmd.pipelineBarrier(
                vk::PipelineStageFlagBits::eTopOfPipe,
                vk::PipelineStageFlagBits::eComputeShader,
                {}, {}, {}, {barrier});
        };

        transitionToGeneral(images.ffxReprojectedRadianceImage);
        transitionToGeneral(images.ffxReprojectedVarianceImage);

        transitionToGeneral(images.ffxPrefilteredRadianceImage);
        transitionToGeneral(images.ffxPrefilteredVarianceImage);
        transitionToGeneral(images.ffxFinalVarianceImage);
        transitionToGeneral(images.ffxAverageRadianceImage);

        for (int i = 0; i < 2; i++) {
            transitionToGeneral(images.ffxHistoryRadianceImages[i]);
            transitionToGeneral(images.ffxHistoryDepthImages[i], vk::ImageAspectFlagBits::eDepth);
            transitionToGeneral(images.ffxHistoryNormalImages[i]);
            transitionToGeneral(images.ffxHistoryRoughnessImages[i]);
            transitionToGeneral(images.ffxSampleCountImages[i]);
        }
        m_initialLayoutTransitionDone = true;
    }

    // Pass 1: Reprojection
    cmd.bindPipeline(vk::PipelineBindPoint::eCompute, *pipelines[0]);
    cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *pipelineLayouts[0], 0, {*descriptorSets[0][pingPongIdx]}, nullptr);
    cmd.dispatch(groupCountX, groupCountY, 1);

    vk::MemoryBarrier memoryBarrier{
        .srcAccessMask = vk::AccessFlagBits::eShaderWrite,
        .dstAccessMask = vk::AccessFlagBits::eShaderRead,
    };
    cmd.pipelineBarrier(
        vk::PipelineStageFlagBits::eComputeShader,
        vk::PipelineStageFlagBits::eComputeShader,
        {}, {memoryBarrier}, {}, {});

    // Pass 2: Prefilter
    cmd.bindPipeline(vk::PipelineBindPoint::eCompute, *pipelines[1]);
    cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *pipelineLayouts[1], 0, {*descriptorSets[1][pingPongIdx]}, nullptr);
    cmd.dispatch(groupCountX, groupCountY, 1);

    cmd.pipelineBarrier(
        vk::PipelineStageFlagBits::eComputeShader,
        vk::PipelineStageFlagBits::eComputeShader,
        {}, {memoryBarrier}, {}, {});

    // Pass 3: Resolve Temporal
    cmd.bindPipeline(vk::PipelineBindPoint::eCompute, *pipelines[2]);
    cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *pipelineLayouts[2], 0, {*descriptorSets[2][pingPongIdx]}, nullptr);
    cmd.dispatch(groupCountX, groupCountY, 1);

    cmd.pipelineBarrier(
        vk::PipelineStageFlagBits::eComputeShader,
        vk::PipelineStageFlagBits::eComputeShader | vk::PipelineStageFlagBits::eFragmentShader | vk::PipelineStageFlagBits::eTransfer, 
        {}, {memoryBarrier}, {}, {});

    // Update History G-Buffer for the NEXT frame
    vk::ImageCopy copyRegionColor{};
    copyRegionColor.srcSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1};
    copyRegionColor.dstSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1};
    copyRegionColor.extent = vk::Extent3D{width, height, 1};

    vk::ImageCopy copyRegionDepth{};
    copyRegionDepth.srcSubresource = {vk::ImageAspectFlagBits::eDepth, 0, 0, 1};
    copyRegionDepth.dstSubresource = {vk::ImageAspectFlagBits::eDepth, 0, 0, 1};
    copyRegionDepth.extent = vk::Extent3D{width, height, 1};

    // Before copy: transition history images from GENERAL to TRANSFER_DST_OPTIMAL
    // We assume src images are in SHADER_READ_ONLY_OPTIMAL or similar, so we transition them to TRANSFER_SRC_OPTIMAL
    // Wait, the caller (main.cc) expects src images to remain in their original layout or we should transition them back.
    // To simplify, we will just transition history images here, and expect the caller to pass them in TRANSFER_SRC_OPTIMAL or we can transition them.
    // Actually, src images are used as descriptors in shader read only optimal. We can just copy them if we transition.
    
    auto transitionImage = [&](vk::Image image, vk::ImageLayout oldLayout, vk::ImageLayout newLayout, vk::AccessFlags srcAccess, vk::AccessFlags dstAccess, vk::PipelineStageFlags srcStage, vk::PipelineStageFlags dstStage, vk::ImageAspectFlags aspect) {
        vk::ImageMemoryBarrier barrier{
            .srcAccessMask = srcAccess,
            .dstAccessMask = dstAccess,
            .oldLayout = oldLayout,
            .newLayout = newLayout,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = image,
            .subresourceRange = {aspect, 0, 1, 0, 1}
        };
        cmd.pipelineBarrier(srcStage, dstStage, {}, {}, {}, {barrier});
    };

    // Transition src to TRANSFER_SRC
    transitionImage(srcDepth, vk::ImageLayout::eShaderReadOnlyOptimal, vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits::eShaderRead, vk::AccessFlagBits::eTransferRead, vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eTransfer, vk::ImageAspectFlagBits::eDepth);
    transitionImage(srcNormal, vk::ImageLayout::eShaderReadOnlyOptimal, vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits::eShaderRead, vk::AccessFlagBits::eTransferRead, vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eTransfer, vk::ImageAspectFlagBits::eColor);
    transitionImage(srcRoughness, vk::ImageLayout::eShaderReadOnlyOptimal, vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits::eShaderRead, vk::AccessFlagBits::eTransferRead, vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eTransfer, vk::ImageAspectFlagBits::eColor);

    // Transition history dst to TRANSFER_DST
    transitionImage(*images.ffxHistoryDepthImages[nextPingPongIdx], vk::ImageLayout::eGeneral, vk::ImageLayout::eTransferDstOptimal, vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eTransferWrite, vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eTransfer, vk::ImageAspectFlagBits::eDepth);
    transitionImage(*images.ffxHistoryNormalImages[nextPingPongIdx], vk::ImageLayout::eGeneral, vk::ImageLayout::eTransferDstOptimal, vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eTransferWrite, vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eTransfer, vk::ImageAspectFlagBits::eColor);
    transitionImage(*images.ffxHistoryRoughnessImages[nextPingPongIdx], vk::ImageLayout::eGeneral, vk::ImageLayout::eTransferDstOptimal, vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eTransferWrite, vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eTransfer, vk::ImageAspectFlagBits::eColor);

    cmd.copyImage(srcDepth, vk::ImageLayout::eTransferSrcOptimal, *images.ffxHistoryDepthImages[nextPingPongIdx], vk::ImageLayout::eTransferDstOptimal, {copyRegionDepth});
    cmd.copyImage(srcNormal, vk::ImageLayout::eTransferSrcOptimal, *images.ffxHistoryNormalImages[nextPingPongIdx], vk::ImageLayout::eTransferDstOptimal, {copyRegionColor});
    cmd.copyImage(srcRoughness, vk::ImageLayout::eTransferSrcOptimal, *images.ffxHistoryRoughnessImages[nextPingPongIdx], vk::ImageLayout::eTransferDstOptimal, {copyRegionColor});

    // Transition src back to SHADER_READ_ONLY
    transitionImage(srcDepth, vk::ImageLayout::eTransferSrcOptimal, vk::ImageLayout::eShaderReadOnlyOptimal, vk::AccessFlagBits::eTransferRead, vk::AccessFlagBits::eShaderRead, vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eFragmentShader | vk::PipelineStageFlagBits::eComputeShader, vk::ImageAspectFlagBits::eDepth);
    transitionImage(srcNormal, vk::ImageLayout::eTransferSrcOptimal, vk::ImageLayout::eShaderReadOnlyOptimal, vk::AccessFlagBits::eTransferRead, vk::AccessFlagBits::eShaderRead, vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eFragmentShader | vk::PipelineStageFlagBits::eComputeShader, vk::ImageAspectFlagBits::eColor);
    transitionImage(srcRoughness, vk::ImageLayout::eTransferSrcOptimal, vk::ImageLayout::eShaderReadOnlyOptimal, vk::AccessFlagBits::eTransferRead, vk::AccessFlagBits::eShaderRead, vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eFragmentShader | vk::PipelineStageFlagBits::eComputeShader, vk::ImageAspectFlagBits::eColor);

    // Transition history dst back to GENERAL
    transitionImage(*images.ffxHistoryDepthImages[nextPingPongIdx], vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eGeneral, vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite, vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eComputeShader, vk::ImageAspectFlagBits::eDepth);
    transitionImage(*images.ffxHistoryNormalImages[nextPingPongIdx], vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eGeneral, vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite, vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eComputeShader, vk::ImageAspectFlagBits::eColor);
    transitionImage(*images.ffxHistoryRoughnessImages[nextPingPongIdx], vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eGeneral, vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite, vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eComputeShader, vk::ImageAspectFlagBits::eColor);
}

void FFXMgr::updateConstantsBuffer(glm::mat4 view, glm::mat4 proj, glm::mat4 prevViewProj) {
    FFXConstants constants{};

    glm::mat4 viewProj = proj * view;
    constants.invViewProjection = glm::inverse(viewProj);
    constants.projection = proj;
    constants.invProjection = glm::inverse(proj);
    constants.prevViewProjection = prevViewProj;
    constants.renderSize[0] = m_width;
    constants.renderSize[1] = m_height;
    constants.temporalStabilityFactor = 0.9f;
    constants.maxSamples = 32;

    void* data = constantsBufferMemory.mapMemory(0, sizeof(FFXConstants));
    std::memcpy(data, &constants, sizeof(FFXConstants));
    constantsBufferMemory.unmapMemory();
}
