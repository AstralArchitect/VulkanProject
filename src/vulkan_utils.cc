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

VulkanUtils::HDRImageData VulkanUtils::loadHDRData(const std::string& filepath) {
    int width = 0;
    int height = 0;
    int channels = 0;

    // Forcer 4 canaux (RGBA float 32-bit) pour assurer la compatibilité avec Vulkan
    float* data = stbi_loadf(filepath.c_str(), &width, &height, &channels, 4);
    if (!data) {
        std::string errStr = stbi_failure_reason() ? stbi_failure_reason() : "Unknown error";
        throw std::runtime_error("Failed to load HDR image: " + filepath + " (" + errStr + ")");
    }

    HDRImageData result;
    result.width = width;
    result.height = height;
    result.channels = 4;
    result.pixels.assign(data, data + (static_cast<size_t>(width) * height * 4));

    stbi_image_free(data);
    return result;
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

VulkanUtils::HDRTexture VulkanUtils::loadHDRTexture(
    const vk::raii::Device& device,
    const vk::raii::PhysicalDevice& physicalDevice,
    const vk::raii::CommandPool& commandPool,
    const vk::raii::Queue& graphicsQueue,
    const std::string& filepath,
    vk::Format format)
{
    HDRImageData hdrData = loadHDRData(filepath);
    vk::DeviceSize imageSize = static_cast<vk::DeviceSize>(hdrData.width) * hdrData.height * 4 * sizeof(float);

    uint32_t mipLevels = static_cast<uint32_t>(std::floor(std::log2(std::max(hdrData.width, hdrData.height)))) + 1;

    // Calcul de la direction du soleil (le pixel le plus lumineux)
    float maxLuminance = -1.0f;
    int maxX = 0;
    int maxY = 0;
    for (int y = 0; y < hdrData.height; ++y) {
        for (int x = 0; x < hdrData.width; ++x) {
            int index = (y * hdrData.width + x) * 4;
            float r = hdrData.pixels[index];
            float g = hdrData.pixels[index + 1];
            float b = hdrData.pixels[index + 2];
            float luminance = 0.2126f * r + 0.7152f * g + 0.0722f * b;
            if (luminance > maxLuminance) {
                maxLuminance = luminance;
                maxX = x;
                maxY = y;
            }
        }
    }

    // Conversion des coordonnées (x,y) en direction 3D pour la projection équirectangulaire
    float u = static_cast<float>(maxX) / static_cast<float>(hdrData.width);
    float v = static_cast<float>(maxY) / static_cast<float>(hdrData.height);
    float theta = (u - 0.5f) * 2.0f * glm::pi<float>();
    float phi = (0.5f - v) * glm::pi<float>();
    float sunY = std::sin(phi);
    float cosPhi = std::cos(phi);
    float sunX = cosPhi * std::cos(theta);
    float sunZ = cosPhi * std::sin(theta);
    glm::vec3 calculatedSunDir = glm::normalize(glm::vec3(sunX, sunY, sunZ));

    auto [stagingBuffer, stagingBufferMemory] = createBuffer(
        device,
        physicalDevice,
        imageSize,
        vk::BufferUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent
    );

    void* mappedData = stagingBufferMemory.mapMemory(0, imageSize);
    std::memcpy(mappedData, hdrData.pixels.data(), static_cast<size_t>(imageSize));
    stagingBufferMemory.unmapMemory();

    auto [image, imageMemory] = createImage(
        device,
        physicalDevice,
        static_cast<uint32_t>(hdrData.width),
        static_cast<uint32_t>(hdrData.height),
        format,
        vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled,
        vk::MemoryPropertyFlagBits::eDeviceLocal,
        vk::SampleCountFlagBits::e1,
        mipLevels
    );

    vk::raii::CommandBuffer cmd = beginSingleTimeCommands(device, commandPool);
    transitionImageLayout(cmd, *image, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal, mipLevels);
    copyBufferToImage(cmd, stagingBuffer, image, static_cast<uint32_t>(hdrData.width), static_cast<uint32_t>(hdrData.height));
    generateMipmaps(physicalDevice, cmd, image, format, hdrData.width, hdrData.height, mipLevels);
    endSingleTimeCommands(std::move(cmd), graphicsQueue);

    vk::raii::ImageView imageView = createImageView(device, *image, format, vk::ImageAspectFlagBits::eColor, mipLevels);

    vk::PhysicalDeviceProperties properties = physicalDevice.getProperties();
    vk::SamplerCreateInfo samplerInfo{
        .magFilter = vk::Filter::eLinear,
        .minFilter = vk::Filter::eLinear,
        .mipmapMode = vk::SamplerMipmapMode::eLinear,
        .addressModeU = vk::SamplerAddressMode::eRepeat,
        .addressModeV = vk::SamplerAddressMode::eRepeat,
        .addressModeW = vk::SamplerAddressMode::eRepeat,
        .mipLodBias = 0.0f,
        .anisotropyEnable = vk::True,
        .maxAnisotropy = properties.limits.maxSamplerAnisotropy,
        .compareEnable = vk::False,
        .compareOp = vk::CompareOp::eAlways,
        .minLod = 0.0f,
        .maxLod = static_cast<float>(mipLevels)};

    vk::raii::Sampler sampler = vk::raii::Sampler(device, samplerInfo);

    HDRTexture texture;
    texture.image = std::move(image);
    texture.imageMemory = std::move(imageMemory);
    texture.imageView = std::move(imageView);
    texture.sampler = std::move(sampler);
    texture.width = static_cast<uint32_t>(hdrData.width);
    texture.height = static_cast<uint32_t>(hdrData.height);
    texture.mipLevels = mipLevels;
    texture.sunDir = calculatedSunDir;

    return texture;
}
