#include "vulkan_utils.hpp"

#include <iostream>
#include <fstream>

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
        vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled,
        vk::MemoryPropertyFlagBits::eDeviceLocal,
        vk::SampleCountFlagBits::e1,
        1
    );

    vk::raii::CommandBuffer cmd = beginSingleTimeCommands(device, commandPool);
    transitionImageLayout(cmd, *image, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal, 1);
    copyBufferToImage(cmd, stagingBuffer, image, static_cast<uint32_t>(hdrData.width), static_cast<uint32_t>(hdrData.height));
    transitionImageLayout(cmd, *image, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal, 1);
    endSingleTimeCommands(std::move(cmd), graphicsQueue);

    vk::raii::ImageView imageView = createImageView(device, *image, format, vk::ImageAspectFlagBits::eColor, 1);

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
        .maxLod = vk::LodClampNone};

    vk::raii::Sampler sampler = vk::raii::Sampler(device, samplerInfo);

    HDRTexture texture;
    texture.image = std::move(image);
    texture.imageMemory = std::move(imageMemory);
    texture.imageView = std::move(imageView);
    texture.sampler = std::move(sampler);
    texture.width = static_cast<uint32_t>(hdrData.width);
    texture.height = static_cast<uint32_t>(hdrData.height);

    return texture;
}
