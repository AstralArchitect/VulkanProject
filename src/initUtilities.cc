#include "vulkan_app.hpp"

#include <iostream>
#include <stdexcept>
#include <cstdlib>
#include <algorithm>
#include <fstream>

#include "vulkan_utils.hpp"

// --- Fonctions libres ---
bool isDeviceSuitable(vk::raii::PhysicalDevice const & physicalDevice, std::vector<const char*> const & requiredDeviceExtension)
{
    auto deviceProperties   = physicalDevice.getProperties();
    auto deviceFeatures     = physicalDevice.getFeatures();

    bool supportsVulkan1_3  = physicalDevice.getProperties().apiVersion >= vk::ApiVersion13;
    auto queueFamilies      = physicalDevice.getQueueFamilyProperties();
    bool supportsGraphics =
        std::ranges::any_of(queueFamilies, [](auto const &qfp) { return !!(qfp.queueFlags & vk::QueueFlagBits::eGraphics); });
    auto availableDeviceExtensions = physicalDevice.enumerateDeviceExtensionProperties();
    bool supportsAllRequiredExtensions =
        std::ranges::all_of(
            requiredDeviceExtension,
            [&availableDeviceExtensions]( auto const & requiredDeviceExtension )
            {
                return std::ranges::any_of(
                    availableDeviceExtensions,
                    [requiredDeviceExtension]( auto const & availableDeviceExtension )
                    { return strcmp( availableDeviceExtension.extensionName, requiredDeviceExtension ) == 0; }
                );
            }
        );
    auto features = physicalDevice.template getFeatures2<vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan11Features, vk::PhysicalDeviceVulkan13Features, vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>();
    bool supportsRequiredFeatures = features.template get<vk::PhysicalDeviceFeatures2>().features.samplerAnisotropy &&
                                    features.template get<vk::PhysicalDeviceFeatures2>().features.independentBlend &&
                                    features.template get<vk::PhysicalDeviceVulkan11Features>().shaderDrawParameters &&
                                    features.template get<vk::PhysicalDeviceVulkan13Features>().dynamicRendering &&
                                    features.template get<vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>().extendedDynamicState;

    // Return true if the physicalDevice meets all the criteria
    return supportsVulkan1_3 && supportsGraphics && supportsAllRequiredExtensions && supportsRequiredFeatures;
}

uint32_t findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties, vk::raii::PhysicalDevice const & physicalDevice) {
    vk::PhysicalDeviceMemoryProperties memProperties = physicalDevice.getMemoryProperties();
    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++)
    {
        if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties)
        {
            return i;
        }
    }

    throw std::runtime_error("failed to find suitable memory type!");
}

static inline uint32_t findMemoryTypeStatic(uint32_t typeFilter, vk::MemoryPropertyFlags properties, vk::raii::PhysicalDevice const & physicalDevice) {
    vk::PhysicalDeviceMemoryProperties memProperties = physicalDevice.getMemoryProperties();
    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++)
    {
        if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties)
        {
            return i;
        }
    }

    throw std::runtime_error("failed to find suitable memory type!");
}

vk::SurfaceFormatKHR VulkanApp::chooseSwapSurfaceFormat(std::vector<vk::SurfaceFormatKHR> const &availableFormats)
{
    assert(!availableFormats.empty());
    const auto formatIt = std::ranges::find_if(availableFormats, [](const auto &format) { return format.format == vk::Format::eB8G8R8A8Srgb && format.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear; });
    return availableFormats[0];
}

vk::PresentModeKHR VulkanApp::chooseSwapPresentMode(std::vector<vk::PresentModeKHR> const &availablePresentModes)
{
    assert(std::ranges::any_of(availablePresentModes, [](auto presentMode) { return presentMode == vk::PresentModeKHR::eFifo; }));
    return std::ranges::any_of(availablePresentModes, [](const vk::PresentModeKHR value) { return vk::PresentModeKHR::eMailbox == value; }) ?
            vk::PresentModeKHR::eMailbox :
            vk::PresentModeKHR::eFifo;
}

vk::Extent2D VulkanApp::chooseSwapExtent(vk::SurfaceCapabilitiesKHR const &capabilities)
{
    if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max())
    {
        return capabilities.currentExtent;
    }
    int width, height;
    glfwGetFramebufferSize(window, &width, &height);

    return {
        std::clamp<uint32_t>(width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width),
        std::clamp<uint32_t>(height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height)
    };
}

uint32_t VulkanApp::chooseSwapMinImageCount(vk::SurfaceCapabilitiesKHR const &surfaceCapabilities)
{
    auto minImageCount = std::max(3u, surfaceCapabilities.minImageCount);
    if ((0 < surfaceCapabilities.maxImageCount) && (surfaceCapabilities.maxImageCount < minImageCount))
    {
        minImageCount = surfaceCapabilities.maxImageCount;
    }
    return minImageCount;
}

vk::Format VulkanApp::findSupportedFormat(const std::vector<vk::Format> &candidates, vk::ImageTiling tiling, vk::FormatFeatureFlags features)
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

vk::Format VulkanApp::findDepthFormat()
{
    return findSupportedFormat({vk::Format::eD32Sfloat, vk::Format::eD32SfloatS8Uint, vk::Format::eD24UnormS8Uint},
                               vk::ImageTiling::eOptimal,
                               vk::FormatFeatureFlagBits::eDepthStencilAttachment);
}

vk::SampleCountFlagBits VulkanApp::getMaxUsableSampleCount() {
    vk::PhysicalDeviceProperties physicalDeviceProperties = physicalDevice.getProperties();

    vk::SampleCountFlags counts = physicalDeviceProperties.limits.framebufferColorSampleCounts & physicalDeviceProperties.limits.framebufferDepthSampleCounts;
    if (counts & vk::SampleCountFlagBits::e64) { return vk::SampleCountFlagBits::e64; }
    if (counts & vk::SampleCountFlagBits::e32) { return vk::SampleCountFlagBits::e32; }
    if (counts & vk::SampleCountFlagBits::e16) { return vk::SampleCountFlagBits::e16; }
    if (counts & vk::SampleCountFlagBits::e8) { return vk::SampleCountFlagBits::e8; }
    if (counts & vk::SampleCountFlagBits::e4) { return vk::SampleCountFlagBits::e4; }
    if (counts & vk::SampleCountFlagBits::e2) { return vk::SampleCountFlagBits::e2; }

    return vk::SampleCountFlagBits::e1;
}

void VulkanApp::createColorResources() {
    vk::Format colorFormat = findSupportedFormat(
        {vk::Format::eR16G16B16A16Sfloat, vk::Format::eR8G8B8A8Unorm, vk::Format::eB8G8R8A8Unorm},
        vk::ImageTiling::eOptimal,
        vk::FormatFeatureFlagBits::eColorAttachment | vk::FormatFeatureFlagBits::eSampledImage);

    vk::ImageUsageFlags usage = vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled;

    std::tie(colorImage, colorImageMemory) = VulkanUtils::createImage(
        device, physicalDevice, swapChainExtent.width, swapChainExtent.height,
        colorFormat, vk::ImageTiling::eOptimal, usage,
        vk::MemoryPropertyFlagBits::eDeviceLocal, msaaSamples);
    colorImageView = VulkanUtils::createImageView(device, *colorImage, colorFormat, vk::ImageAspectFlagBits::eColor);
}