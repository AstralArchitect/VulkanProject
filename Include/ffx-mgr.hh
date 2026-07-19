#pragma once

#include <cstddef>
#if defined(__INTELLISENSE__) || !defined(USE_CPP20_MODULES)
#define VULKAN_HPP_NO_STRUCT_CONSTRUCTORS
#include <vulkan/vulkan_raii.hpp>
#else
#define VULKAN_HPP_NO_STRUCT_CONSTRUCTORS
import vulkan_hpp;
#endif

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/hash.hpp>

// AMD FidelityFX Denoiser
// -----------------------
struct FfxImages {
    // FFX Denoiser Intermediate Buffers
    vk::raii::Image        ffxReprojectedRadianceImage = nullptr;
    vk::raii::DeviceMemory ffxReprojectedRadianceMemory = nullptr;
    vk::raii::ImageView    ffxReprojectedRadianceView = nullptr;

    vk::raii::Image        ffxReprojectedVarianceImage = nullptr;
    vk::raii::DeviceMemory ffxReprojectedVarianceMemory = nullptr;
    vk::raii::ImageView    ffxReprojectedVarianceView = nullptr;

    vk::raii::Image        ffxSampleCountImages[2] = {nullptr, nullptr};
    vk::raii::DeviceMemory ffxSampleCountMemory[2] = {nullptr, nullptr};
    vk::raii::ImageView    ffxSampleCountViews[2] = {nullptr, nullptr};

    vk::raii::Image        ffxAverageRadianceImage = nullptr;
    vk::raii::DeviceMemory ffxAverageRadianceMemory = nullptr;
    vk::raii::ImageView    ffxAverageRadianceView = nullptr;

    vk::raii::Image        ffxPrefilteredRadianceImage = nullptr;
    vk::raii::DeviceMemory ffxPrefilteredRadianceMemory = nullptr;
    vk::raii::ImageView    ffxPrefilteredRadianceView = nullptr;

    vk::raii::Image        ffxPrefilteredVarianceImage = nullptr;
    vk::raii::DeviceMemory ffxPrefilteredVarianceMemory = nullptr;
    vk::raii::ImageView    ffxPrefilteredVarianceView = nullptr;

    vk::raii::Image        ffxFinalVarianceImage = nullptr;
    vk::raii::DeviceMemory ffxFinalVarianceMemory = nullptr;
    vk::raii::ImageView    ffxFinalVarianceView = nullptr;

    // FFX Denoiser History Buffers (Ping-Pong)
    vk::raii::Image        ffxHistoryRadianceImages[2] = {nullptr, nullptr};
    vk::raii::DeviceMemory ffxHistoryRadianceMemory[2] = {nullptr, nullptr};
    vk::raii::ImageView    ffxHistoryRadianceViews[2] = {nullptr, nullptr};

    vk::raii::Image        ffxHistoryDepthImages[2] = {nullptr, nullptr};
    vk::raii::DeviceMemory ffxHistoryDepthMemory[2] = {nullptr, nullptr};
    vk::raii::ImageView    ffxHistoryDepthViews[2] = {nullptr, nullptr};

    vk::raii::Image        ffxHistoryNormalImages[2] = {nullptr, nullptr};
    vk::raii::DeviceMemory ffxHistoryNormalMemory[2] = {nullptr, nullptr};
    vk::raii::ImageView    ffxHistoryNormalViews[2] = {nullptr, nullptr};

    vk::raii::Image        ffxHistoryRoughnessImages[2] = {nullptr, nullptr};
    vk::raii::DeviceMemory ffxHistoryRoughnessMemory[2] = {nullptr, nullptr};
    vk::raii::ImageView    ffxHistoryRoughnessViews[2] = {nullptr, nullptr};
};

struct FFXConstants {
    glm::mat4       invViewProjection;
    glm::mat4       projection;
    glm::mat4       invProjection;
    glm::mat4       prevViewProjection;
    unsigned int    renderSize[2];
    float           temporalStabilityFactor;
    int             maxSamples;
};

class FFXMgr {
public:
    FFXMgr(vk::raii::Device &device, vk::raii::PhysicalDevice &physicalDevice, vk::raii::CommandPool &commandPool, uint32_t width, uint32_t height);
    FfxImages images;
    uint32_t m_width;
    uint32_t m_height;

    void updateReprojDescriptorSets(
        vk::ImageView inputRadiance,
        vk::ImageView inputDepth,
        vk::ImageView inputNormal,
        vk::ImageView inputRoughness,
        vk::ImageView inputMotionVectors,
        vk::ImageView inputRayLength);

    void updatePrefilDescriptorSets(
        vk::ImageView inputRadiance,
        vk::ImageView inputDepth,
        vk::ImageView inputNormal,
        vk::ImageView inputRoughness);

    void updateResolvDescriptorSets(vk::ImageView inputRoughness);

    void dispatchDenoiser(vk::raii::CommandBuffer& cmd, uint32_t frameIndex, uint32_t width, uint32_t height, vk::Image srcDepth, vk::Image srcNormal, vk::Image srcRoughness);

    void updateConstantsBuffer(glm::mat4 view, glm::mat4 proj, glm::mat4 prevViewProj);
private:
    bool m_initialLayoutTransitionDone = false;
    vk::raii::Device* device;
    vk::raii::PhysicalDevice* physicalDevice;
    vk::raii::CommandPool* commandPool;

    vk::raii::Buffer        constantsBuffer       = nullptr;
    vk::raii::DeviceMemory  constantsBufferMemory = nullptr;
    std::vector<void*>      constantsBuffersMapped;

    vk::raii::Sampler linearSampler = nullptr;

    vk::raii::DescriptorPool                            descriptorPool = nullptr;
    std::array<vk::raii::DescriptorSetLayout, 3>        descriptorSetLayouts = {nullptr, nullptr, nullptr};
    std::array<vk::raii::PipelineLayout, 3>             pipelineLayouts = {nullptr, nullptr, nullptr};
    std::array<vk::raii::Pipeline, 3>                   pipelines = {nullptr, nullptr, nullptr};
    std::array<std::vector<vk::raii::DescriptorSet>, 3> descriptorSets;

    void createDescriptorPool();

    void createReprojShaderResources();
    void createPrefilShaderResources();
    void createResolvShaderResources();
};