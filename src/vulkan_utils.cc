#include "vulkan_utils.hpp"

#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>

#include "stb_image.h"
#include "vulkan/vulkan_raii.hpp"

[[nodiscard]] vk::raii::ShaderModule VulkanUtils::createShaderModule(const std::vector<char>& code, const vk::raii::Device &device)
{
    vk::ShaderModuleCreateInfo createInfo{ .codeSize = code.size() * sizeof(char), .pCode = reinterpret_cast<const uint32_t*>(code.data()) };
    vk::raii::ShaderModule shaderModule{ device, createInfo };
    return shaderModule;
}

std::vector<char> VulkanUtils::readFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);

    if (!file.is_open()) {
        throw std::runtime_error("failed to open file!");
    }

    std::vector<char> buffer(file.tellg());

    file.seekg(0, std::ios::beg);
    file.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));

    file.close();

    return buffer;
}

void VulkanUtils::generateMipmaps(
    const vk::raii::PhysicalDevice &physicalDevice,
    vk::raii::CommandBuffer &commandBuffer,
    vk::raii::Image &image,
    vk::Format imageFormat,
    int32_t texWidth,
    int32_t texHeight,
    uint32_t mipLevels)
{
    vk::FormatProperties formatProperties = physicalDevice.getFormatProperties(imageFormat);

    if (!(formatProperties.optimalTilingFeatures & vk::FormatFeatureFlagBits::eSampledImageFilterLinear))
    {
        throw std::runtime_error("texture image format does not support linear blitting!");
    }

    vk::ImageMemoryBarrier barrier = {
        .srcAccessMask = vk::AccessFlagBits::eTransferWrite,
        .dstAccessMask = vk::AccessFlagBits::eTransferRead,
        .oldLayout = vk::ImageLayout::eTransferDstOptimal,
        .newLayout = vk::ImageLayout::eTransferSrcOptimal,
        .srcQueueFamilyIndex = vk::QueueFamilyIgnored,
        .dstQueueFamilyIndex = vk::QueueFamilyIgnored,
        .image = image,
        .subresourceRange = {
            .aspectMask = vk::ImageAspectFlagBits::eColor,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1}};

    int32_t mipWidth = texWidth;
    int32_t mipHeight = texHeight;

    for (uint32_t i = 1; i < mipLevels; i++)
    {
        barrier.subresourceRange.baseMipLevel = i - 1;
        barrier.oldLayout = vk::ImageLayout::eTransferDstOptimal;
        barrier.newLayout = vk::ImageLayout::eTransferSrcOptimal;
        barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
        barrier.dstAccessMask = vk::AccessFlagBits::eTransferRead;

        commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eTransfer, {}, {}, {}, barrier);

        vk::ImageBlit blit = {
            .srcSubresource = {.aspectMask = vk::ImageAspectFlagBits::eColor, .mipLevel = i - 1, .layerCount = 1},
            .srcOffsets = std::array<vk::Offset3D, 2>({{}, {mipWidth, mipHeight, 1}}),
            .dstSubresource = {.aspectMask = vk::ImageAspectFlagBits::eColor, .mipLevel = i, .layerCount = 1},
            .dstOffsets = std::array<vk::Offset3D, 2>({{}, {1 < mipWidth ? mipWidth / 2 : 1, 1 < mipHeight ? mipHeight / 2 : 1, 1}})};

        commandBuffer.blitImage(image, vk::ImageLayout::eTransferSrcOptimal, image, vk::ImageLayout::eTransferDstOptimal, blit, vk::Filter::eLinear);

        barrier.oldLayout = vk::ImageLayout::eTransferSrcOptimal;
        barrier.newLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
        barrier.srcAccessMask = vk::AccessFlagBits::eTransferRead;
        barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;

        commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eFragmentShader, {}, {}, {}, barrier);

        if (1 < mipWidth)
        {
            mipWidth /= 2;
        }
        if (1 < mipHeight)
        {
            mipHeight /= 2;
        }
    }

    barrier.subresourceRange.baseMipLevel = mipLevels - 1;
    barrier.oldLayout = vk::ImageLayout::eTransferDstOptimal;
    barrier.newLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
    barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
    barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;

    commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eFragmentShader, {}, {}, {}, barrier);
}

void VulkanUtils::BackgroundTexture::create(
    const vk::raii::Device& device,
    const vk::raii::PhysicalDevice& physicalDevice,
    const vk::raii::CommandPool& commandPool,
    const vk::raii::Queue& graphicsQueue,
    const vk::raii::DescriptorSetLayout& cameraSetLayout,
    uint32_t cubeMapSize,
    vk::Format format,
    const std::string& shaderSpvPath)
{
    width = cubeMapSize;
    height = cubeMapSize;
    mipLevels = static_cast<uint32_t>(std::floor(std::log2(cubeMapSize))) + 1;

    // 1. Creation de l'image Vulkan compatible Cubemap (arrayLayers = 6, eCubeCompatible)
    vk::ImageCreateInfo imageInfo{};
    imageInfo.flags = vk::ImageCreateFlagBits::eCubeCompatible;
    imageInfo.imageType = vk::ImageType::e2D;
    imageInfo.format = format;
    imageInfo.extent = vk::Extent3D{cubeMapSize, cubeMapSize, 1};
    imageInfo.mipLevels = mipLevels;
    imageInfo.arrayLayers = 6;
    imageInfo.samples = vk::SampleCountFlagBits::e1;
    imageInfo.tiling = vk::ImageTiling::eOptimal;
    imageInfo.usage = vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst;
    imageInfo.sharingMode = vk::SharingMode::eExclusive;
    imageInfo.initialLayout = vk::ImageLayout::eUndefined;

    image = device.createImage(imageInfo);

    // Allocation memoire Device Local
    vk::MemoryRequirements memRequirements = image.getMemoryRequirements();
    uint32_t memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal, physicalDevice);

    vk::MemoryAllocateInfo allocInfo{
        .allocationSize = memRequirements.size,
        .memoryTypeIndex = memoryTypeIndex
    };
    imageMemory = device.allocateMemory(allocInfo);
    image.bindMemory(*imageMemory, 0);

    // 2. Transiter l'image vers Layout eGeneral pour que le Compute Shader puisse ecrire dedans
    vk::raii::CommandBuffer cmd = beginSingleTimeCommands(device, commandPool);
    vk::ImageMemoryBarrier barrier{
        .srcAccessMask = vk::AccessFlagBits::eNone,
        .dstAccessMask = vk::AccessFlagBits::eShaderWrite,
        .oldLayout = vk::ImageLayout::eUndefined,
        .newLayout = vk::ImageLayout::eGeneral,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = *image,
        .subresourceRange = vk::ImageSubresourceRange{
            .aspectMask = vk::ImageAspectFlagBits::eColor,
            .baseMipLevel = 0,
            .levelCount = mipLevels,
            .baseArrayLayer = 0,
            .layerCount = 6
        }
    };

    cmd.pipelineBarrier(
        vk::PipelineStageFlagBits::eTopOfPipe,
        vk::PipelineStageFlagBits::eComputeShader,
        vk::DependencyFlags{},
        nullptr, nullptr, barrier
    );
    endSingleTimeCommands(std::move(cmd), graphicsQueue);

    // 3. Creation de l'ImageView principale en type eCube (6 faces) pour le sampling dans les shaders
    vk::ImageViewCreateInfo viewInfo{
        .image = *image,
        .viewType = vk::ImageViewType::eCube,
        .format = format,
        .subresourceRange = vk::ImageSubresourceRange{
            .aspectMask = vk::ImageAspectFlagBits::eColor,
            .baseMipLevel = 0,
            .levelCount = mipLevels,
            .baseArrayLayer = 0,
            .layerCount = 6
        }
    };
    imageView = device.createImageView(viewInfo);

    // 4. Creation du Sampler adapte aux Cubemaps
    vk::PhysicalDeviceProperties properties = physicalDevice.getProperties();
    vk::SamplerCreateInfo samplerInfo{
        .magFilter = vk::Filter::eLinear,
        .minFilter = vk::Filter::eLinear,
        .mipmapMode = vk::SamplerMipmapMode::eLinear,
        .addressModeU = vk::SamplerAddressMode::eClampToEdge,
        .addressModeV = vk::SamplerAddressMode::eClampToEdge,
        .addressModeW = vk::SamplerAddressMode::eClampToEdge,
        .mipLodBias = 0.0f,
        .anisotropyEnable = vk::True,
        .maxAnisotropy = properties.limits.maxSamplerAnisotropy,
        .compareEnable = vk::False,
        .compareOp = vk::CompareOp::eAlways,
        .minLod = 0.0f,
        .maxLod = static_cast<float>(mipLevels)
    };
    sampler = device.createSampler(samplerInfo);

    // 5. Creation des ImageViews individuelles par Mip Level (Storage 2DArray, 6 layers)
    mipViews.clear();
    for (uint32_t level = 0; level < mipLevels; ++level) {
        vk::ImageViewCreateInfo mipViewInfo{
            .image = *image,
            .viewType = vk::ImageViewType::e2DArray,
            .format = format,
            .subresourceRange = vk::ImageSubresourceRange{
                .aspectMask = vk::ImageAspectFlagBits::eColor,
                .baseMipLevel = level,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 6
            }
        };
        mipViews.push_back(device.createImageView(mipViewInfo));
    }

    // 6. Descriptor Set Layout pour le Compute (Set 1, Binding 0: Storage Image)
    vk::DescriptorSetLayoutBinding storageBinding{
        .binding = 0,
        .descriptorType = vk::DescriptorType::eStorageImage,
        .descriptorCount = 1,
        .stageFlags = vk::ShaderStageFlagBits::eCompute
    };
    vk::DescriptorSetLayoutCreateInfo computeLayoutInfo{
        .bindingCount = 1,
        .pBindings = &storageBinding
    };
    computeDescriptorSetLayout = device.createDescriptorSetLayout(computeLayoutInfo);

    // Push Constants Range (16 octets)
    vk::PushConstantRange pushConstantRange{
        .stageFlags = vk::ShaderStageFlagBits::eCompute,
        .offset = 0,
        .size = sizeof(uint32_t) * 3 + sizeof(float)
    };

    // Set 0 = cameraSetLayout, Set 1 = computeDescriptorSetLayout
    std::array<vk::DescriptorSetLayout, 2> setLayouts = { *cameraSetLayout, *computeDescriptorSetLayout };
    vk::PipelineLayoutCreateInfo pipelineLayoutInfo{
        .setLayoutCount = static_cast<uint32_t>(setLayouts.size()),
        .pSetLayouts = setLayouts.data(),
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &pushConstantRange
    };
    computePipelineLayout = device.createPipelineLayout(pipelineLayoutInfo);

    // 7. Pipeline de Compute
    vk::raii::ShaderModule computeShaderModule = VulkanUtils::createShaderModule(VulkanUtils::readFile(shaderSpvPath), device);
    vk::ComputePipelineCreateInfo computePipelineInfo{
        .stage = vk::PipelineShaderStageCreateInfo{
            .stage = vk::ShaderStageFlagBits::eCompute,
            .module = *computeShaderModule,
            .pName = "main"
        },
        .layout = *computePipelineLayout
    };
    computePipeline = device.createComputePipeline(nullptr, computePipelineInfo);

    // 8. Descriptor Pool & Sets pour chaque Mip Level
    vk::DescriptorPoolSize poolSize{
        .type = vk::DescriptorType::eStorageImage,
        .descriptorCount = mipLevels
    };
    vk::DescriptorPoolCreateInfo poolInfo{
        .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
        .maxSets = mipLevels,
        .poolSizeCount = 1,
        .pPoolSizes = &poolSize
    };
    computeDescriptorPool = device.createDescriptorPool(poolInfo);

    std::vector<vk::DescriptorSetLayout> layouts(mipLevels, *computeDescriptorSetLayout);
    vk::DescriptorSetAllocateInfo allocSetInfo{
        .descriptorPool = *computeDescriptorPool,
        .descriptorSetCount = mipLevels,
        .pSetLayouts = layouts.data()
    };
    computeDescriptorSets = device.allocateDescriptorSets(allocSetInfo);

    // Mise a jour des Descriptor Sets avec les ImageViews par mip level
    for (uint32_t level = 0; level < mipLevels; ++level) {
        vk::DescriptorImageInfo storageImageInfo{
            .imageView = *mipViews[level],
            .imageLayout = vk::ImageLayout::eGeneral
        };
        vk::WriteDescriptorSet writeSet{
            .dstSet = *computeDescriptorSets[level],
            .dstBinding = 0,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eStorageImage,
            .pImageInfo = &storageImageInfo
        };
        device.updateDescriptorSets({ writeSet }, nullptr);
    }
}

void VulkanUtils::BackgroundTexture::update(
    const vk::raii::CommandBuffer& cmd,
    const vk::raii::DescriptorSet& cameraDescriptorSet)
{
    // Barriere initiale : Transiter vers eGeneral pour que le Compute Shader puisse ecrire
    vk::ImageMemoryBarrier preBarrier{
        .srcAccessMask = vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eNone,
        .dstAccessMask = vk::AccessFlagBits::eShaderWrite,
        .oldLayout = vk::ImageLayout::eUndefined,
        .newLayout = vk::ImageLayout::eGeneral,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = *image,
        .subresourceRange = vk::ImageSubresourceRange{
            .aspectMask = vk::ImageAspectFlagBits::eColor,
            .baseMipLevel = 0,
            .levelCount = mipLevels,
            .baseArrayLayer = 0,
            .layerCount = 6
        }
    };

    cmd.pipelineBarrier(
        vk::PipelineStageFlagBits::eTopOfPipe | vk::PipelineStageFlagBits::eFragmentShader,
        vk::PipelineStageFlagBits::eComputeShader,
        vk::DependencyFlags{},
        nullptr, nullptr, preBarrier
    );

    cmd.bindPipeline(vk::PipelineBindPoint::eCompute, *computePipeline);
    cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *computePipelineLayout, 0, *cameraDescriptorSet, nullptr);

    for (uint32_t level = 0; level < mipLevels; ++level) {
        uint32_t currentMipSize = std::max(1u, width >> level);
        float roughness = (mipLevels > 1) ? (static_cast<float>(level) / static_cast<float>(mipLevels - 1)) : 0.0f;

        struct PushConstants {
            uint32_t faceSize;
            uint32_t mipLevel;
            uint32_t maxMipLevel;
            float roughness;
        } push{
            .faceSize = currentMipSize,
            .mipLevel = level,
            .maxMipLevel = mipLevels,
            .roughness = roughness
        };

        cmd.pushConstants<PushConstants>(*computePipelineLayout, vk::ShaderStageFlagBits::eCompute, 0, push);
        cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *computePipelineLayout, 1, *computeDescriptorSets[level], nullptr);

        uint32_t groupCountX = (currentMipSize + 15) / 16;
        uint32_t groupCountY = (currentMipSize + 15) / 16;
        cmd.dispatch(groupCountX, groupCountY, 6);
    }

    // Barriere pour passer la Cubemap en eShaderReadOnlyOptimal pour le Fragment/RT Shader
    vk::ImageMemoryBarrier barrier{
        .srcAccessMask = vk::AccessFlagBits::eShaderWrite,
        .dstAccessMask = vk::AccessFlagBits::eShaderRead,
        .oldLayout = vk::ImageLayout::eGeneral,
        .newLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = *image,
        .subresourceRange = vk::ImageSubresourceRange{
            .aspectMask = vk::ImageAspectFlagBits::eColor,
            .baseMipLevel = 0,
            .levelCount = mipLevels,
            .baseArrayLayer = 0,
            .layerCount = 6
        }
    };

    cmd.pipelineBarrier(
        vk::PipelineStageFlagBits::eComputeShader,
        vk::PipelineStageFlagBits::eFragmentShader,
        vk::DependencyFlags{},
        nullptr, nullptr, barrier
    );
}

VulkanUtils::BackgroundTexture VulkanUtils::createProceduralSkyCubemap(
    const vk::raii::Device& device,
    const vk::raii::PhysicalDevice& physicalDevice,
    const vk::raii::CommandPool& commandPool,
    const vk::raii::Queue& graphicsQueue,
    const vk::raii::DescriptorSetLayout& cameraSetLayout,
    uint32_t cubeMapSize,
    vk::Format format)
{
    BackgroundTexture texture;
    texture.create(device, physicalDevice, commandPool, graphicsQueue, cameraSetLayout, cubeMapSize, format);
    return texture;
}
