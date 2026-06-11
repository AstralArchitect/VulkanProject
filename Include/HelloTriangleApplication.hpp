#pragma once

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
#include <array>
#include <glm/glm.hpp>

// --- Constantes et variables globales ---
static constexpr uint32_t WIDTH = 800;
static constexpr uint32_t HEIGHT = 600;
static constexpr int MAX_FRAMES_IN_FLIGHT = 2;

struct Vertex
{
    glm::vec2 pos;
    glm::vec3 color;

	static vk::VertexInputBindingDescription getBindingDescription()
    {
      return {.binding = 0, .stride = sizeof(Vertex), .inputRate = vk::VertexInputRate::eVertex};
    }
	static std::array<vk::VertexInputAttributeDescription, 2> getAttributeDescriptions()
    {
      return {{{.location = 0, .binding = 0, .format = vk::Format::eR32G32Sfloat, .offset = offsetof(Vertex, pos)},
               {.location = 1, .binding = 0, .format = vk::Format::eR32G32B32Sfloat, .offset = offsetof(Vertex, color)}}};
    }
};

extern const std::vector<Vertex> vertices;

class HelloTriangleApplication {
public:
    bool framebufferResized = false;
    void run();

private:
    std::vector<const char*> requiredDeviceExtension = {vk::KHRSwapchainExtensionName, "VK_EXT_extended_dynamic_state"};
    
    GLFWwindow* window = nullptr;
    
    vk::raii::Context  context;
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

    vk::raii::PipelineLayout pipelineLayout = nullptr;
    vk::raii::Pipeline graphicsPipeline = nullptr;

    vk::raii::CommandPool commandPool = nullptr;
    std::vector<vk::raii::CommandBuffer> commandBuffers;

    vk::raii::Buffer vertexBuffer = nullptr;
    vk::raii::DeviceMemory vertexBufferMemory = nullptr;

    std::vector<vk::raii::Semaphore> presentCompleteSemaphores;
    std::vector<vk::raii::Semaphore> renderFinishedSemaphores;
    std::vector<vk::raii::Fence> inFlightFences;

    uint32_t frameIndex = 0;

    // Méthodes d'initialisation
    void initWindow();
    void initVulkan();
    void mainLoop();
    void cleanup();

    // Méthodes Vulkan
    void createInstance();
    void setupDebugMessenger();
    void populateDebugMessengerCreateInfo(vk::DebugUtilsMessengerCreateInfoEXT& createInfo);
    void createSurface();
    void pickPhysicalDevice();
    void createLogicalDevice();
    void createCommandPool();
    void createCommandBuffers();
    void recordCommandBuffer(uint32_t imageIndex);
    void transition_image_layout(uint32_t imageIndex, vk::ImageLayout old_layout, vk::ImageLayout new_layout, vk::AccessFlags2 src_access_mask, vk::AccessFlags2 dst_access_mask, vk::PipelineStageFlags2 src_stage_mask, vk::PipelineStageFlags2 dst_stage_mask);
    void drawFrame();
    void createSyncObjects();
    void createVertexBuffer();

    void recreateSwapChain();
    void cleanupSwapChain();

    void createBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage, vk::MemoryPropertyFlags properties, vk::raii::Buffer& buffer, vk::raii::DeviceMemory& bufferMemory);
    void copyBuffer(vk::raii::Buffer & srcBuffer, vk::raii::Buffer & dstBuffer, vk::DeviceSize size);
    
    // Méthodes utilitaires
    vk::SurfaceFormatKHR chooseSwapSurfaceFormat(std::vector<vk::SurfaceFormatKHR> const &availableFormats);
    vk::PresentModeKHR chooseSwapPresentMode(std::vector<vk::PresentModeKHR> const &availablePresentModes);
    vk::Extent2D chooseSwapExtent(vk::SurfaceCapabilitiesKHR const &capabilities);
    uint32_t chooseSwapMinImageCount(vk::SurfaceCapabilitiesKHR const &surfaceCapabilities);

    [[nodiscard]] vk::raii::ShaderModule createShaderModule(const std::vector<char>& code) const;
    
    void createSwapChain();
    void createImageViews();
    void createGraphicsPipeline();

    static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
        VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
        VkDebugUtilsMessageTypeFlagsEXT messageType,
        const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
        void* pUserData);
};