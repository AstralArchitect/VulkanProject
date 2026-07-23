#pragma once

#include "vulkan_utils.hpp"
#if defined(__INTELLISENSE__) || !defined(USE_CPP20_MODULES)
#define VULKAN_HPP_NO_STRUCT_CONSTRUCTORS
#include <vulkan/vulkan_raii.hpp>
#else
#define VULKAN_HPP_NO_STRUCT_CONSTRUCTORS
import vulkan_hpp;
#endif

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <vector>

#ifndef STB_IMAGE
#include "stb_image.h"
#endif

#ifndef TINYGLTF
#include <tiny_gltf.h>
#endif

#include "model.hpp"

#include "text_manager.hpp"

#include "camera.hpp"

#include "ffx-mgr.hh"

#include "physics_world.hpp"

// --- Constantes et variables globales ---
static constexpr uint32_t WIDTH = 800;
static constexpr uint32_t HEIGHT = 600;
extern const int MAX_FRAMES_IN_FLIGHT;

const std::string MODEL_PATH = "res/models/horloge.glb";
const std::string TEXTURE_PATH = "res/textures/viking_room.png";

class VulkanApp
{
public:
    bool framebufferResized = false;

    void init();
    void run();
private:
    // structs 
    struct PhysicsEntity {
        GltfModel* graphicModel;
        JPH::BodyID physicsBodyId;
    };

    struct CameraUBO {
        alignas(16) glm::mat4 view;
        alignas(16) glm::mat4 proj;
        alignas(16) glm::mat4 prevViewProj;
        alignas(16) glm::vec4 camPos;        // vec4
        alignas(16) float time;
    };

    std::vector<const char *> requiredDeviceExtension = {
        vk::KHRSwapchainExtensionName, 
        "VK_EXT_extended_dynamic_state", 
        "VK_EXT_vertex_input_dynamic_state",
        "VK_KHR_acceleration_structure",
        "VK_KHR_ray_query",
        "VK_KHR_deferred_host_operations"};

    GLFWwindow *window = nullptr;

    vk::raii::Context context;
    vk::raii::Instance instance = nullptr;
    vk::raii::DebugUtilsMessengerEXT debugMessenger = nullptr;

    vk::raii::PhysicalDevice physicalDevice = nullptr;
    vk::raii::Device device = nullptr;

    vk::raii::Queue graphicsQueue = nullptr;
    uint32_t queueIndex = ~0U;

    vk::raii::SurfaceKHR surface = nullptr;

    vk::Extent2D swapChainExtent;
    vk::SurfaceFormatKHR swapChainSurfaceFormat;
    vk::raii::SwapchainKHR swapChain = nullptr;
    std::vector<vk::Image> swapChainImages;

    std::vector<vk::raii::ImageView> swapChainImageViews;

    vk::raii::DescriptorSetLayout descriptorSetLayout = nullptr;
    vk::raii::PipelineLayout pipelineLayout = nullptr;
    vk::raii::Pipeline graphicsPipeline = nullptr;
    vk::raii::Pipeline backgroundPipeline = nullptr;

    vk::raii::CommandPool commandPool = nullptr;
    std::vector<vk::raii::CommandBuffer> commandBuffers;

    std::vector<std::unique_ptr<GltfModel>> models;
    std::vector<PhysicsEntity> physicsEntities;

    TextureManager textureManager;

    vk::raii::Image depthImage = nullptr;
    vk::raii::DeviceMemory depthImageMemory = nullptr;
    vk::raii::ImageView depthImageView = nullptr;

    // Render images (reflect, normals, roughness, motion vecotr, ray length)
    vk::raii::Image renderImages[5] = {nullptr, nullptr, nullptr, nullptr, nullptr};
    vk::raii::DeviceMemory renderImagesMemory[5] = {nullptr, nullptr, nullptr, nullptr, nullptr};
    vk::raii::ImageView renderImagesView[5] = {nullptr, nullptr, nullptr, nullptr, nullptr};

    vk::raii::DescriptorPool             compositionDescriptorPool = nullptr;
    vk::raii::DescriptorSetLayout        compositionDescriptorSetLayout = nullptr;
    vk::raii::PipelineLayout             compositionPipelineLayout = nullptr;
    vk::raii::Pipeline                   compositionPipeline = nullptr;
    std::vector<vk::raii::DescriptorSet> compositionDescriptorSets;

    vk::raii::Image colorImage = nullptr;
    vk::raii::DeviceMemory colorImageMemory = nullptr;
    vk::raii::ImageView colorImageView = nullptr;

    VulkanUtils::HDRTexture backgroundTexture;

    Camera camera;

    float deltaTime;
    float lastFrame;

    std::vector<vk::raii::Buffer> cameraUniformBuffers;
    std::vector<vk::raii::DeviceMemory> cameraUniformBuffersMemory;
    std::vector<void*> cameraUniformBuffersMapped;

    vk::raii::DescriptorPool descriptorPool = nullptr;
    std::vector<vk::raii::DescriptorSet> cameraDescriptorSets;

    vk::SampleCountFlagBits msaaSamples = vk::SampleCountFlagBits::e1;

    std::vector<vk::raii::Semaphore> presentCompleteSemaphores;
    std::vector<vk::raii::Semaphore> renderFinishedSemaphores;
    std::vector<vk::raii::Fence> inFlightFences;

    vk::raii::Buffer tlasBuffer = nullptr;
    vk::raii::DeviceMemory tlasBufferMemory = nullptr;
    vk::raii::Buffer tlasScratchBuffer = nullptr;
    vk::raii::DeviceMemory tlasScratchBufferMemory = nullptr;
    vk::raii::AccelerationStructureKHR tlasHandle = nullptr;
    vk::raii::Buffer instancesBuffer = nullptr;
    vk::raii::DeviceMemory instancesBufferMemory = nullptr;
    void* instancesBufferMapped = nullptr;
    uint32_t blasInstancesCount;

    vk::raii::Buffer instanceDataBuffer = nullptr;
    vk::raii::DeviceMemory instanceDataBufferMemory = nullptr;
    void* instanceDataBufferMapped = nullptr;

    uint32_t frameIndex = 0;

    FFXMgr* ffxMgr = nullptr;

    std::unique_ptr<PhysicsWorld> physicsWorld;

    // Méthodes d'initialisation
    void initWindow();
    void initVulkan();
    void mainLoop();
    void cleanup();

    void processInput(GLFWwindow *window);
    void mouse(double xposIn, double yposIn);

    static void mouseCallback(GLFWwindow* window, double xpos, double ypos) {
        auto app = static_cast<VulkanApp*>(glfwGetWindowUserPointer(window));
        if (app) {
            app->mouse(xpos, ypos);
        }
    }

    // Méthodes Vulkan
    void createInstance();
    void setupDebugMessenger();
    void populateDebugMessengerCreateInfo(vk::DebugUtilsMessengerCreateInfoEXT &createInfo);

    void createSurface();
    void pickPhysicalDevice();
    void createLogicalDevice();

    void createSwapChain();
    void createImageViews();
    void createDescriptorSetLayout();
    void createGraphicsPipeline();
    void createBackgroundPipeline();
    
    void createCommandPool();
    void createCommandBuffers();
    
    void transition_image_layout(uint32_t imageIndex, vk::ImageLayout old_layout, vk::ImageLayout new_layout, vk::AccessFlags2 src_access_mask, vk::AccessFlags2 dst_access_mask, vk::PipelineStageFlags2 src_stage_mask, vk::PipelineStageFlags2 dst_stage_mask, vk::ImageAspectFlags image_aspect_flags, uint32_t mipLevels = 1);
    void transition_image_layout(const vk::raii::Image *image, vk::ImageLayout old_layout, vk::ImageLayout new_layout, vk::AccessFlags2 src_access_mask, vk::AccessFlags2 dst_access_mask, vk::PipelineStageFlags2 src_stage_mask, vk::PipelineStageFlags2 dst_stage_mask, vk::ImageAspectFlags image_aspect_flags, uint32_t mipLevels = 1);
    
    void createSyncObjects();
    void createUniformBuffers();

    void createBackgroundTexture();

    void createDescriptorPool();
    void createDescriptorSets();
    
    void createDepthResources();

    void createRenderResources();
    void createCompositionResources();
    void updateCompositionDescriptorSet(uint32_t imageIndex, uint32_t frameIndex);

    void initFfx();

    void loadModels();

    void createTlas();
    void updateTlasInstances();

    void drawFrame();
    void updateUniformBuffer(uint32_t currentImage);
    void recordCommandBuffer(uint32_t imageIndex);

    void recreateSwapChain();
    void cleanupSwapChain();

    // Méthodes utilitaires
    vk::SurfaceFormatKHR    chooseSwapSurfaceFormat(std::vector<vk::SurfaceFormatKHR> const &availableFormats);
    vk::PresentModeKHR      chooseSwapPresentMode(std::vector<vk::PresentModeKHR> const &availablePresentModes);
    vk::Extent2D            chooseSwapExtent(vk::SurfaceCapabilitiesKHR const &capabilities);
    uint32_t                chooseSwapMinImageCount(vk::SurfaceCapabilitiesKHR const &surfaceCapabilities);
   
    vk::Format              findSupportedFormat(const std::vector<vk::Format> &candidates, vk::ImageTiling tiling, vk::FormatFeatureFlags features);
    vk::Format              findDepthFormat();

    vk::SampleCountFlagBits getMaxUsableSampleCount();

    void generateMipmaps(vk::raii::CommandBuffer &commandBuffer, vk::raii::Image &image, vk::Format imageFormat, int32_t texWidth, int32_t texHeight, uint32_t mipLevels);
    void createColorResources();

    [[nodiscard]] vk::raii::ShaderModule createShaderModule(const std::vector<char> &code) const;

    static VKAPI_ATTR VkBool32 VKAPI_CALL
    debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, VkDebugUtilsMessageTypeFlagsEXT messageType, const VkDebugUtilsMessengerCallbackDataEXT *pCallbackData, void *pUserData);
};