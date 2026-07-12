#pragma once

#if defined(__INTELLISENSE__) || !defined(USE_CPP20_MODULES)
#define VULKAN_HPP_NO_STRUCT_CONSTRUCTORS
#include <vulkan/vulkan_raii.hpp>
#else
#define VULKAN_HPP_NO_STRUCT_CONSTRUCTORS
import vulkan_hpp;
#endif

#include <stdexcept>
#include <utility>
#include <vector>

namespace VulkanUtils {

    // Find a memory type that fits the properties and type filter
    inline uint32_t findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties, vk::raii::PhysicalDevice const & physicalDevice) {
        vk::PhysicalDeviceMemoryProperties memProperties = physicalDevice.getMemoryProperties();
        for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
            if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
                return i;
            }
        }
        throw std::runtime_error("failed to find suitable memory type!");
    }

    // Create a Vulkan buffer and allocate memory for it
    inline std::pair<vk::raii::Buffer, vk::raii::DeviceMemory> createBuffer(
        vk::raii::Device const &device,
        vk::raii::PhysicalDevice const &physicalDevice,
        vk::DeviceSize size,
        vk::BufferUsageFlags usage,
        vk::MemoryPropertyFlags properties)
    {
        vk::BufferCreateInfo bufferInfo{
            .size = size,
            .usage = usage,
            .sharingMode = vk::SharingMode::eExclusive
        };
        vk::raii::Buffer buffer(device, bufferInfo);

        vk::MemoryRequirements memRequirements = buffer.getMemoryRequirements();
        vk::MemoryAllocateInfo allocInfo{
            .allocationSize = memRequirements.size,
            .memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties, physicalDevice)
        };
        vk::raii::DeviceMemory bufferMemory(device, allocInfo);

        buffer.bindMemory(*bufferMemory, 0);

        return {std::move(buffer), std::move(bufferMemory)};
    }

    // Begin a one-time command buffer
    inline vk::raii::CommandBuffer beginSingleTimeCommands(
        vk::raii::Device const &device,
        vk::raii::CommandPool const &commandPool)
    {
        vk::CommandBufferAllocateInfo allocInfo{
            .commandPool = *commandPool,
            .level = vk::CommandBufferLevel::ePrimary,
            .commandBufferCount = 1
        };
        vk::raii::CommandBuffer commandBuffer = std::move(vk::raii::CommandBuffers(device, allocInfo).front());

        vk::CommandBufferBeginInfo beginInfo{.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit};
        commandBuffer.begin(beginInfo);

        return commandBuffer;
    }

    // End and submit a one-time command buffer
    inline void endSingleTimeCommands(
        vk::raii::CommandBuffer &&commandBuffer,
        vk::raii::Queue const &graphicsQueue)
    {
        commandBuffer.end();

        vk::SubmitInfo submitInfo{.commandBufferCount = 1, .pCommandBuffers = &*commandBuffer};
        graphicsQueue.submit(submitInfo, nullptr);
        graphicsQueue.waitIdle();
    }

    // Copy buffer data from source to destination
    inline void copyBuffer(
        vk::raii::Device const &device,
        vk::raii::CommandPool const &commandPool,
        vk::raii::Queue const &graphicsQueue,
        vk::raii::Buffer &srcBuffer,
        vk::raii::Buffer &dstBuffer,
        vk::DeviceSize size)
    {
        vk::raii::CommandBuffer commandCopyBuffer = beginSingleTimeCommands(device, commandPool);
        commandCopyBuffer.copyBuffer(*srcBuffer, *dstBuffer, vk::BufferCopy{.size = size});
        endSingleTimeCommands(std::move(commandCopyBuffer), graphicsQueue);
    }

    // Create a Vulkan image and allocate memory for it
    inline std::pair<vk::raii::Image, vk::raii::DeviceMemory> createImage(
        vk::raii::Device const &device,
        vk::raii::PhysicalDevice const &physicalDevice,
        uint32_t width,
        uint32_t height,
        vk::Format format,
        vk::ImageTiling tiling,
        vk::ImageUsageFlags usage,
        vk::MemoryPropertyFlags properties,
        vk::SampleCountFlagBits numSamples,
        uint32_t mipLevels = 1)
    {
        vk::ImageCreateInfo imageInfo{
            .imageType = vk::ImageType::e2D,
            .format = format,
            .extent = {width, height, 1},
            .mipLevels = mipLevels,
            .arrayLayers = 1,
            .samples = numSamples,
            .tiling = tiling,
            .usage = usage,
            .sharingMode = vk::SharingMode::eExclusive
        };

        vk::raii::Image image(device, imageInfo);

        vk::MemoryRequirements memRequirements = image.getMemoryRequirements();
        vk::MemoryAllocateInfo allocInfo{
            .allocationSize = memRequirements.size,
            .memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties, physicalDevice)
        };
        vk::raii::DeviceMemory imageMemory(device, allocInfo);
        image.bindMemory(*imageMemory, 0);

        return {std::move(image), std::move(imageMemory)};
    }

    // Create an image view
    inline vk::raii::ImageView createImageView(
        vk::raii::Device const &device,
        vk::Image const &image,
        vk::Format format,
        vk::ImageAspectFlags aspectFlags,
        uint32_t mipLevels = 1)
    {
        vk::ImageViewCreateInfo viewInfo{
            .image = image,
            .viewType = vk::ImageViewType::e2D,
            .format = format,
            .subresourceRange = {
                .aspectMask = aspectFlags,
                .baseMipLevel = 0,
                .levelCount = mipLevels,
                .baseArrayLayer = 0,
                .layerCount = 1
            }
        };
        return vk::raii::ImageView(device, viewInfo);
    }

    // Free function to transition an image layout (standard pipeline barrier)
    inline void transitionImageLayout(
        vk::raii::CommandBuffer &commandBuffer,
        vk::Image const &image,
        vk::ImageLayout oldLayout,
        vk::ImageLayout newLayout,
        uint32_t mipLevels = 1)
    {
        vk::ImageMemoryBarrier barrier{
            .oldLayout = oldLayout,
            .newLayout = newLayout,
            .srcQueueFamilyIndex = vk::QueueFamilyIgnored,
            .dstQueueFamilyIndex = vk::QueueFamilyIgnored,
            .image = image,
            .subresourceRange = {
                .aspectMask = vk::ImageAspectFlagBits::eColor,
                .baseMipLevel = 0,
                .levelCount = mipLevels,
                .baseArrayLayer = 0,
                .layerCount = 1
            }
        };

        vk::PipelineStageFlags sourceStage;
        vk::PipelineStageFlags destinationStage;

        if (oldLayout == vk::ImageLayout::eUndefined && newLayout == vk::ImageLayout::eTransferDstOptimal)
        {
            barrier.srcAccessMask = vk::AccessFlags{};
            barrier.dstAccessMask = vk::AccessFlagBits::eTransferWrite;

            sourceStage = vk::PipelineStageFlagBits::eTopOfPipe;
            destinationStage = vk::PipelineStageFlagBits::eTransfer;
        }
        else if (oldLayout == vk::ImageLayout::eTransferDstOptimal && newLayout == vk::ImageLayout::eShaderReadOnlyOptimal)
        {
            barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
            barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;

            sourceStage = vk::PipelineStageFlagBits::eTransfer;
            destinationStage = vk::PipelineStageFlagBits::eFragmentShader;
        }
        else
        {
            throw std::invalid_argument("unsupported layout transition!");
        }

        commandBuffer.pipelineBarrier(sourceStage, destinationStage, {}, {}, {}, barrier);
    }

    // Find supported format from candidates
    inline vk::Format findSupportedFormat(
        vk::raii::PhysicalDevice const &physicalDevice,
        const std::vector<vk::Format> &candidates,
        vk::ImageTiling tiling,
        vk::FormatFeatureFlags features)
    {
        for (const auto format : candidates)
        {
            vk::FormatProperties props = physicalDevice.getFormatProperties(format);
            if (
                ((tiling == vk::ImageTiling::eLinear) && ((props.linearTilingFeatures & features) == features)) ||
                ((tiling == vk::ImageTiling::eOptimal) && ((props.optimalTilingFeatures & features) == features)))
            {
                return format;
            }
        }
        throw std::runtime_error("failed to find supported format!");
    }

    inline void copyBufferToImage(vk::raii::CommandBuffer &commandBuffer, const vk::raii::Buffer &buffer, vk::raii::Image &image, uint32_t width, uint32_t height)
    {
        vk::BufferImageCopy region{.bufferOffset = 0,
                                .bufferRowLength = 0,
                                .bufferImageHeight = 0,
                                .imageSubresource = {.aspectMask = vk::ImageAspectFlagBits::eColor,
                                                        .mipLevel = 0,
                                                        .baseArrayLayer = 0,
                                                        .layerCount = 1},
                                .imageOffset = {0, 0, 0},
                                .imageExtent = {width, height, 1}};

        commandBuffer.copyBufferToImage(buffer, image, vk::ImageLayout::eTransferDstOptimal, region);
    }
}
