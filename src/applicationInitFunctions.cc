#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <glm/gtc/quaternion.hpp>

#include "vulkan_app.hpp"
#include "ffx-mgr.hh"

#include "vulkan_utils.hpp"

#include "physics_world.hpp"

const std::vector<char const *> validationLayers = {"VK_LAYER_KHRONOS_validation"};

#ifdef NDEBUG
constexpr bool enableValidationLayers = false;
#else
constexpr bool enableValidationLayers = true;
#endif

#ifdef _WIN32
#include <windows.h>

#include "logic_engine.hh"

#include "sun_calc.hh"

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_vulkan.h"

void sleep_ms(DWORD milliseconds)
{
    Sleep(milliseconds);
}

#else

#include <time.h>

void sleep_ms(unsigned long milliseconds)
{
    struct timespec ts;

    ts.tv_sec = milliseconds / 1000;
    ts.tv_nsec = (milliseconds % 1000) * 1000000;

    nanosleep(&ts, NULL);
}

#endif

bool isDeviceSuitable(vk::raii::PhysicalDevice const &physicalDevice, std::vector<const char *> const &requiredDeviceExtension);
std::vector<char> readFile(const std::string &filename);

static void framebufferResizeCallback(GLFWwindow *window, int width, int height)
{
    auto app = reinterpret_cast<VulkanApp *>(glfwGetWindowUserPointer(window));
    app->framebufferResized = true;
}

void VulkanApp::init(LogicEngine* logic)
{
    logicEngine = logic;
    initWindow();
    PhysicsWorld::global_init();
    physicsWorld = PhysicsWorld::create();
    initVulkan();
}

void VulkanApp::run()
{
    mainLoop();
    cleanup();
}

// process all input: query GLFW whether relevant keys are pressed/released this frame and react accordingly
// ---------------------------------------------------------------------------------------------------------
void VulkanApp::processInput(GLFWwindow *window)
{
    static bool isFullscreen = false;
    static int windowedWidth, windowedHeight, windowedPosX, windowedPosY;

    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);

    if (logicEngine) {
        logicEngine->updatePlayerMovement(window, physicsWorld.get(), deltaTime);
    }

    if (glfwGetKey(window, GLFW_KEY_F11) == GLFW_PRESS)
    {
        if (!isFullscreen)
        {
            glfwGetWindowSize(window, &windowedWidth, &windowedHeight);
            glfwGetWindowPos(window, &windowedPosX, &windowedPosY);

            const GLFWvidmode *mode = glfwGetVideoMode(glfwGetPrimaryMonitor());
            glfwSetWindowMonitor(window, glfwGetPrimaryMonitor(), 0, 0, mode->width, mode->height, mode->refreshRate);
        }
        else
        {
            glfwSetWindowMonitor(window, NULL, windowedPosX, windowedPosY, windowedWidth, windowedHeight, 0);
        }
        isFullscreen = !isFullscreen;
        sleep_ms(100);
    }

    static bool tabPressedLastFrame = false;
    bool tabPressed = glfwGetKey(window, GLFW_KEY_TAB) == GLFW_PRESS;

    if (tabPressed && !tabPressedLastFrame) {
        uiMode = !uiMode;
        if (uiMode) {
            glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
        } else {
            glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
        }
    }
    tabPressedLastFrame = tabPressed;
}

// Mouse callback (Logique FPV / First Person View)
void VulkanApp::mouse(double xposIn, double yposIn) {
    if (uiMode) return;

    static double lastX = WIDTH / 2.0;
    static double lastY = HEIGHT / 2.0;
    static bool firstMouse = true;

    if (firstMouse)
    {
        lastX = xposIn;
        lastY = yposIn;
        firstMouse = false;
    }

    float xoffset = xposIn - lastX;
    float yoffset = lastY - yposIn; // inversé car les coordonnées Y vont du bas vers le haut
    lastX = xposIn;
    lastY = yposIn;

    camera->ProcessMouseMovement(xoffset, yoffset);
}


void VulkanApp::initWindow()
{
    if (!glfwInit())
    {
        throw std::runtime_error("Échec de l'initialisation de GLFW");
    }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

    window = glfwCreateWindow(WIDTH, HEIGHT, "RT App", nullptr, nullptr);
    glfwSetWindowUserPointer(window, this);
    glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
    glfwSetCursorPosCallback(window, mouseCallback);
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    if (!window)
    {
        throw std::runtime_error("Échec de la création de la fenêtre GLFW");
    }
}

void VulkanApp::initVulkan()
{
    createInstance();
    setupDebugMessenger();
    createSurface();
    pickPhysicalDevice();
    createLogicalDevice();

    createSwapChain();
    createImageViews();

    createCommandPool();
    createCommandBuffers();

    textureManager.init(device, physicalDevice, commandPool, graphicsQueue);
    skinMgr = std::make_unique<SkinMgr>(device, physicalDevice);

    createDescriptorSetLayout();
    createGraphicsPipeline();
    createBackgroundPipeline();

    loadModels();
    createTlas();

    createUniformBuffers();
    createColorResources();
    createDepthResources();
    createRenderResources();
    initFfx();
    createCompositionResources();
    createBackgroundTexture();

    createDescriptorPool();
    createDescriptorSets();
    createSyncObjects();

    initImGui();
}

void VulkanApp::createInstance()
{
    vk::ApplicationInfo appInfo{};

    appInfo.pApplicationName = "RT App";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "No Engine";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = vk::ApiVersion14;

    // Get the required instance extensions from GLFW.
    uint32_t glfwExtensionCount = 0;
    auto glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
    std::vector<const char *> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

    if (enableValidationLayers)
    {
        extensions.push_back(vk::EXTDebugUtilsExtensionName);
    }

    // Check if the required extensions are supported.
    auto extensionProperties = context.enumerateInstanceExtensionProperties();
    for (auto const &extension : extensions)
    {
        if (std::ranges::none_of(extensionProperties, [extension](auto const &extensionProperty)
                                 { return strcmp(extensionProperty.extensionName, extension) == 0; }))
        {
            throw std::runtime_error("Required extension not supported: " + std::string(extension));
        }
    }

    vk::InstanceCreateInfo createInfo{};
    createInfo.pApplicationInfo = &appInfo;
    createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    createInfo.ppEnabledExtensionNames = extensions.data();

    vk::DebugUtilsMessengerCreateInfoEXT debugCreateInfo{};
    if (enableValidationLayers)
    {
        auto layerProperties = context.enumerateInstanceLayerProperties();
        for (const char *layerName : validationLayers)
        {
            if (std::ranges::none_of(layerProperties, [layerName](auto const &layerProperty)
                                     { return strcmp(layerProperty.layerName, layerName) == 0; }))
            {
                throw std::runtime_error("Validation layer requested, but not available: " + std::string(layerName));
            }
        }
        createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
        createInfo.ppEnabledLayerNames = validationLayers.data();

        populateDebugMessengerCreateInfo(debugCreateInfo);
        createInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT *)&debugCreateInfo;
    }

    instance = vk::raii::Instance(context, createInfo);
}

void VulkanApp::populateDebugMessengerCreateInfo(vk::DebugUtilsMessengerCreateInfoEXT &createInfo)
{
    createInfo.messageSeverity = vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose |
                                 vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
                                 vk::DebugUtilsMessageSeverityFlagBitsEXT::eError;
    createInfo.messageType = vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral |
                             vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation |
                             vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance;
    createInfo.pfnUserCallback = reinterpret_cast<vk::PFN_DebugUtilsMessengerCallbackEXT>(debugCallback);
}

void VulkanApp::setupDebugMessenger()
{
    if (!enableValidationLayers)
        return;

    vk::DebugUtilsMessengerCreateInfoEXT createInfo{};
    populateDebugMessengerCreateInfo(createInfo);

    debugMessenger = vk::raii::DebugUtilsMessengerEXT(instance, createInfo);
}

VKAPI_ATTR VkBool32 VKAPI_CALL VulkanApp::debugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, VkDebugUtilsMessageTypeFlagsEXT messageType,
    const VkDebugUtilsMessengerCallbackDataEXT *pCallbackData, void *pUserData)
{
    std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;

    return VK_FALSE;
}

void VulkanApp::pickPhysicalDevice()
{
    std::vector<vk::raii::PhysicalDevice> physicalDevices = instance.enumeratePhysicalDevices();

    bool deviceFound = false;

    for (auto &device : physicalDevices)
    {
        if (isDeviceSuitable(device, requiredDeviceExtension))
        {
            physicalDevice = std::move(device);
            msaaSamples = vk::SampleCountFlagBits::e1;
            deviceFound = true;
            break;
        }
    }

    if (!deviceFound)
    {
        throw std::runtime_error("failed to find a suitable GPU!");
    }
}

void VulkanApp::createLogicalDevice()
{
    std::vector<vk::QueueFamilyProperties> queueFamilyProperties = physicalDevice.getQueueFamilyProperties();

    for (uint32_t qfpIndex = 0; qfpIndex < queueFamilyProperties.size(); qfpIndex++)
    {
        if ((queueFamilyProperties[qfpIndex].queueFlags & vk::QueueFlagBits::eGraphics) &&
            physicalDevice.getSurfaceSupportKHR(qfpIndex, *surface))
        {
            // found a queue family that supports both graphics and present
            queueIndex = qfpIndex;
            break;
        }
    }
    if (queueIndex == ~0U)
    {
        throw std::runtime_error("Could not find a queue for graphics and present -> terminating");
    }

    auto graphicsQueueFamilyProperty = std::ranges::find_if(queueFamilyProperties, [](auto const &qfp)
                                                            { return (qfp.queueFlags & vk::QueueFlagBits::eGraphics) != static_cast<vk::QueueFlags>(0); });
    auto graphicsIndex =
        static_cast<uint32_t>(std::distance(queueFamilyProperties.begin(), graphicsQueueFamilyProperty));

    float queuePriority = 0.5f;
    vk::DeviceQueueCreateInfo deviceQueueCreateInfo{};
    deviceQueueCreateInfo.queueFamilyIndex = graphicsIndex;
    deviceQueueCreateInfo.queueCount = 1;
    deviceQueueCreateInfo.pQueuePriorities = &queuePriority;

    // Create a chain of feature structures
    vk::StructureChain<
        vk::PhysicalDeviceFeatures2,
        vk::PhysicalDeviceVulkan11Features,
        vk::PhysicalDeviceVulkan12Features,
        vk::PhysicalDeviceVulkan13Features,
        vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT,
        vk::PhysicalDeviceVertexInputDynamicStateFeaturesEXT,
        vk::PhysicalDeviceAccelerationStructureFeaturesKHR,
        vk::PhysicalDeviceRayQueryFeaturesKHR>
        featureChain = {
            {.features = {
                 .independentBlend = true,
                 .samplerAnisotropy = true,
                 .shaderSampledImageArrayDynamicIndexing = true,
                 .shaderInt16 = true}},
            {.shaderDrawParameters = true},
            {.shaderFloat16 = true, .descriptorBindingPartiallyBound = true, .runtimeDescriptorArray = false, .bufferDeviceAddress = true},
            {.synchronization2 = true, .dynamicRendering = true},
            {.extendedDynamicState = true},
            {.vertexInputDynamicState = true},
            {.accelerationStructure = true},
            {.rayQuery = true}};

    vk::DeviceCreateInfo deviceCreateInfo{
        .pNext = &featureChain.get<vk::PhysicalDeviceFeatures2>(),
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &deviceQueueCreateInfo,
        .enabledExtensionCount = static_cast<uint32_t>(requiredDeviceExtension.size()),
        .ppEnabledExtensionNames = requiredDeviceExtension.data()};

    device = vk::raii::Device(physicalDevice, deviceCreateInfo);
    graphicsQueue = vk::raii::Queue(device, graphicsIndex, 0);
}

void VulkanApp::createSurface()
{
    VkSurfaceKHR _surface;
    if (glfwCreateWindowSurface(*instance, window, nullptr, &_surface) != 0)
    {
        throw std::runtime_error("failed to create window surface!");
    }
    surface = vk::raii::SurfaceKHR(instance, _surface);
}

void VulkanApp::createSwapChain()
{
    vk::SurfaceCapabilitiesKHR surfaceCapabilities = physicalDevice.getSurfaceCapabilitiesKHR(*surface);
    swapChainExtent = chooseSwapExtent(surfaceCapabilities);
    uint32_t minImageCount = chooseSwapMinImageCount(surfaceCapabilities);

    std::vector<vk::SurfaceFormatKHR> availableFormats = physicalDevice.getSurfaceFormatsKHR(*surface);
    swapChainSurfaceFormat = chooseSwapSurfaceFormat(availableFormats);

    std::vector<vk::PresentModeKHR> availablePresentModes = physicalDevice.getSurfacePresentModesKHR(*surface);

    vk::SwapchainCreateInfoKHR swapChainCreateInfo{
        .surface = *surface,
        .minImageCount = minImageCount,
        .imageFormat = swapChainSurfaceFormat.format,
        .imageColorSpace = swapChainSurfaceFormat.colorSpace,
        .imageExtent = swapChainExtent,
        .imageArrayLayers = 1,
        .imageUsage = vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eStorage,
        .imageSharingMode = vk::SharingMode::eExclusive,
        .preTransform = surfaceCapabilities.currentTransform,
        .compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque,
        .presentMode = chooseSwapPresentMode(availablePresentModes),
        .clipped = true};

    swapChain = vk::raii::SwapchainKHR(device, swapChainCreateInfo);
    swapChainImages = swapChain.getImages();
}

void VulkanApp::createImageViews()
{
    assert(swapChainImageViews.empty());
    vk::ImageViewCreateInfo imageViewCreateInfo{.viewType = vk::ImageViewType::e2D,
                                                .format = swapChainSurfaceFormat.format,
                                                .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}};

    imageViewCreateInfo.components = {vk::ComponentSwizzle::eIdentity, vk::ComponentSwizzle::eIdentity,
                                      vk::ComponentSwizzle::eIdentity, vk::ComponentSwizzle::eIdentity};

    imageViewCreateInfo.subresourceRange = {
        .aspectMask = vk::ImageAspectFlagBits::eColor, .levelCount = 1, .layerCount = 1};

    for (auto &image : swapChainImages)
    {
        imageViewCreateInfo.image = image;
        swapChainImageViews.emplace_back(device, imageViewCreateInfo);
    }
}

void VulkanApp::createDescriptorSetLayout()
{
    std::array global_bindings = {
        vk::DescriptorSetLayoutBinding(0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment | vk::ShaderStageFlagBits::eCompute, nullptr),
        vk::DescriptorSetLayoutBinding(1, vk::DescriptorType::eAccelerationStructureKHR, 1, vk::ShaderStageFlagBits::eFragment, nullptr),
        vk::DescriptorSetLayoutBinding(2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eFragment, nullptr),
        vk::DescriptorSetLayoutBinding(3, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment, nullptr),
    };

    std::array<vk::DescriptorSetLayoutBinding, global_bindings.size()> bindings = global_bindings;

    vk::DescriptorSetLayoutCreateInfo layoutInfo{.bindingCount = static_cast<uint32_t>(bindings.size()), .pBindings = bindings.data()};
    descriptorSetLayout = vk::raii::DescriptorSetLayout(device, layoutInfo);
}

void VulkanApp::initFfx() {
    ffxMgr = new FFXMgr(device, physicalDevice, commandPool, swapChainExtent.width, swapChainExtent.height);

    vk::CommandBufferAllocateInfo allocInfo{.commandPool = *commandPool,
                                            .level = vk::CommandBufferLevel::ePrimary,
                                            .commandBufferCount = 1};
    vk::raii::CommandBuffer cmd = std::move(device.allocateCommandBuffers(allocInfo).front());
    cmd.begin({.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

    auto transition_to_general = [&](const vk::raii::Image& image, vk::ImageAspectFlags aspect = vk::ImageAspectFlagBits::eColor) {
        vk::ImageMemoryBarrier2 barrier = {
            .srcStageMask = vk::PipelineStageFlagBits2::eTopOfPipe,
            .srcAccessMask = {},
            .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
            .dstAccessMask = vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite,
            .oldLayout = vk::ImageLayout::eUndefined,
            .newLayout = vk::ImageLayout::eGeneral,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = *image,
            .subresourceRange = {.aspectMask = aspect,
                                 .baseMipLevel = 0, .levelCount = 1,
                                 .baseArrayLayer = 0, .layerCount = 1}};
        vk::DependencyInfo dependency_info = {
            .dependencyFlags = {}, .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &barrier};
        cmd.pipelineBarrier2(dependency_info);
    };

    transition_to_general(ffxMgr->images.ffxReprojectedRadianceImage);
    transition_to_general(ffxMgr->images.ffxReprojectedVarianceImage);

    transition_to_general(ffxMgr->images.ffxPrefilteredRadianceImage);
    transition_to_general(ffxMgr->images.ffxPrefilteredVarianceImage);
    transition_to_general(ffxMgr->images.ffxFinalVarianceImage);
    transition_to_general(ffxMgr->images.ffxAverageRadianceImage);

    for (int i = 0; i < 2; i++) {
        transition_to_general(ffxMgr->images.ffxHistoryRadianceImages[i]);
        transition_to_general(ffxMgr->images.ffxHistoryDepthImages[i], vk::ImageAspectFlagBits::eDepth);
        transition_to_general(ffxMgr->images.ffxHistoryNormalImages[i]);
        transition_to_general(ffxMgr->images.ffxHistoryRoughnessImages[i]);
        transition_to_general(ffxMgr->images.ffxSampleCountImages[i]);
    }

    cmd.end();
    vk::SubmitInfo submitInfo{.commandBufferCount = 1, .pCommandBuffers = &*cmd};
    graphicsQueue.submit(submitInfo);
    graphicsQueue.waitIdle();

    //                                 reflection                           normal               roughness            motion vector        ray length
    ffxMgr->updateReprojDescriptorSets(renderImagesView[0], depthImageView, renderImagesView[1], renderImagesView[2], renderImagesView[3], renderImagesView[4]);
    ffxMgr->updatePrefilDescriptorSets(renderImagesView[0], depthImageView, renderImagesView[1], renderImagesView[2]);
    ffxMgr->updateResolvDescriptorSets(renderImagesView[2]);
}

void VulkanApp::createGraphicsPipeline()
{
    vk::raii::ShaderModule shaderModule = VulkanUtils::createShaderModule(VulkanUtils::readFile("builddir/shader.spv"), device);

    vk::PipelineShaderStageCreateInfo vertShaderStageInfo{
        .stage = vk::ShaderStageFlagBits::eVertex, .module = shaderModule, .pName = "vertMain"};

    vk::PipelineShaderStageCreateInfo fragShaderStageInfo{
        .stage = vk::ShaderStageFlagBits::eFragment, .module = shaderModule, .pName = "fragMain"};

    vk::PipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo, fragShaderStageInfo};

    std::vector<vk::DynamicState> dynamicStates = {vk::DynamicState::eViewport, vk::DynamicState::eScissor, vk::DynamicState::eVertexInputEXT};

    vk::PipelineDynamicStateCreateInfo dynamicState{
        .dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()),
        .pDynamicStates = dynamicStates.data()};

    vk::PipelineVertexInputStateCreateInfo vertexInputInfo{};

    vk::PipelineInputAssemblyStateCreateInfo inputAssembly{.topology = vk::PrimitiveTopology::eTriangleList};

    vk::PipelineViewportStateCreateInfo viewportState{.viewportCount = 1, .scissorCount = 1};

    vk::PipelineRasterizationStateCreateInfo rasterizer{
        .depthClampEnable = vk::False,
        .rasterizerDiscardEnable = vk::False,
        .polygonMode = vk::PolygonMode::eFill,
        .cullMode = vk::CullModeFlagBits::eNone,
        .frontFace = vk::FrontFace::eCounterClockwise,
        .depthBiasEnable = vk::False,
        .lineWidth = 1.0f};

    vk::PipelineMultisampleStateCreateInfo multisampling{
        .rasterizationSamples = msaaSamples,
        .sampleShadingEnable = vk::False};

    std::array<vk::PipelineColorBlendAttachmentState, 6> colorBlendAttachments;
    for (auto &attachment : colorBlendAttachments)
    {
        attachment.blendEnable = vk::False;
        attachment.colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
                                    vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA;
    }
    vk::PipelineColorBlendStateCreateInfo colorBlending{
        .logicOpEnable = vk::False,
        .logicOp = vk::LogicOp::eCopy,
        .attachmentCount = static_cast<uint32_t>(colorBlendAttachments.size()),
        .pAttachments = colorBlendAttachments.data()};

    vk::PipelineDepthStencilStateCreateInfo depthStencil{
        .depthTestEnable = vk::True,
        .depthWriteEnable = vk::True,
        .depthCompareOp = vk::CompareOp::eLess,
        .depthBoundsTestEnable = vk::False,
        .stencilTestEnable = vk::False};

    vk::PushConstantRange pushConstantRange{
        .stageFlags = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
        .offset = 0,
        .size = sizeof(MeshPushConstants)};

    std::array<vk::DescriptorSetLayout, 2> layouts = {
        *descriptorSetLayout,                    // Set 0 (Caméra)
        *textureManager.getDescriptorSetLayout() // Set 1 (Textures)
    };
    
    vk::PipelineLayoutCreateInfo pipelineLayoutInfo{
        .setLayoutCount = static_cast<uint32_t>(layouts.size()),
        .pSetLayouts = layouts.data(),
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &pushConstantRange};

    vk::Format renderFormat = findSupportedFormat(
        {vk::Format::eR16G16B16A16Sfloat, vk::Format::eR8G8B8A8Unorm, vk::Format::eB8G8R8A8Unorm},
        vk::ImageTiling::eOptimal,
        vk::FormatFeatureFlagBits::eColorAttachment | vk::FormatFeatureFlagBits::eSampledImage);

    std::array<vk::Format, 6> colorFormats = {
        renderFormat,
        renderFormat,
        renderFormat,
        renderFormat,
        renderFormat,
        renderFormat
    };

    pipelineLayout = vk::raii::PipelineLayout(device, pipelineLayoutInfo);

    vk::StructureChain<vk::GraphicsPipelineCreateInfo, vk::PipelineRenderingCreateInfo> pipelineCreateInfoChain = {
        {.stageCount = 2,
         .pStages = shaderStages,
         .pVertexInputState = &vertexInputInfo,
         .pInputAssemblyState = &inputAssembly,
         .pViewportState = &viewportState,
         .pRasterizationState = &rasterizer,
         .pMultisampleState = &multisampling,
         .pDepthStencilState = &depthStencil,
         .pColorBlendState = &colorBlending,
         .pDynamicState = &dynamicState,
         .layout = pipelineLayout,
         .renderPass = nullptr},
        {.colorAttachmentCount = static_cast<uint32_t>(colorFormats.size()),
         .pColorAttachmentFormats = colorFormats.data(),
         .depthAttachmentFormat = findDepthFormat()}};

    graphicsPipeline =
        vk::raii::Pipeline(device, nullptr, pipelineCreateInfoChain.get<vk::GraphicsPipelineCreateInfo>());
}

void VulkanApp::createBackgroundPipeline()
{
    vk::raii::ShaderModule shaderModule = VulkanUtils::createShaderModule(VulkanUtils::readFile("builddir/background.spv"), device);

    vk::PipelineShaderStageCreateInfo vertShaderStageInfo{
        .stage = vk::ShaderStageFlagBits::eVertex, .module = shaderModule, .pName = "vertMain"};

    vk::PipelineShaderStageCreateInfo fragShaderStageInfo{
        .stage = vk::ShaderStageFlagBits::eFragment, .module = shaderModule, .pName = "fragMain"};

    vk::PipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo, fragShaderStageInfo};

    std::vector<vk::DynamicState> dynamicStates = {vk::DynamicState::eViewport, vk::DynamicState::eScissor};

    vk::PipelineDynamicStateCreateInfo dynamicState{
        .dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()),
        .pDynamicStates = dynamicStates.data()};

    vk::PipelineVertexInputStateCreateInfo vertexInputInfo{};

    vk::PipelineInputAssemblyStateCreateInfo inputAssembly{.topology = vk::PrimitiveTopology::eTriangleList};

    vk::PipelineViewportStateCreateInfo viewportState{.viewportCount = 1, .scissorCount = 1};

    vk::PipelineRasterizationStateCreateInfo rasterizer{
        .depthClampEnable = vk::False,
        .rasterizerDiscardEnable = vk::False,
        .polygonMode = vk::PolygonMode::eFill,
        .cullMode = vk::CullModeFlagBits::eNone,
        .frontFace = vk::FrontFace::eCounterClockwise,
        .depthBiasEnable = vk::False,
        .lineWidth = 1.0f};

    vk::PipelineMultisampleStateCreateInfo multisampling{
        .rasterizationSamples = msaaSamples,
        .sampleShadingEnable = vk::False};

    std::array<vk::PipelineColorBlendAttachmentState, 6> colorBlendAttachments;
    for (int i = 0; i < 6; i++)
    {
        colorBlendAttachments[i].blendEnable = vk::False;
        if (i == 0) {
            colorBlendAttachments[i].colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
                                                      vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA;
        } else {
            colorBlendAttachments[i].colorWriteMask = {};
        }
    }
    
    vk::PipelineColorBlendStateCreateInfo colorBlending{
        .logicOpEnable = vk::False,
        .logicOp = vk::LogicOp::eCopy,
        .attachmentCount = static_cast<uint32_t>(colorBlendAttachments.size()),
        .pAttachments = colorBlendAttachments.data()};

    vk::PipelineDepthStencilStateCreateInfo depthStencil{
        .depthTestEnable = vk::True,
        .depthWriteEnable = vk::False,
        .depthCompareOp = vk::CompareOp::eLessOrEqual, // Z = 1.0 passes
        .depthBoundsTestEnable = vk::False,
        .stencilTestEnable = vk::False};

    vk::Format renderFormat = findSupportedFormat(
        {vk::Format::eR16G16B16A16Sfloat, vk::Format::eR8G8B8A8Unorm, vk::Format::eB8G8R8A8Unorm},
        vk::ImageTiling::eOptimal,
        vk::FormatFeatureFlagBits::eColorAttachment | vk::FormatFeatureFlagBits::eSampledImage);

    std::array<vk::Format, 6> colorFormats = {
        renderFormat, renderFormat, renderFormat, renderFormat, renderFormat, renderFormat
    };

    vk::StructureChain<vk::GraphicsPipelineCreateInfo, vk::PipelineRenderingCreateInfo> pipelineCreateInfoChain = {
        {.stageCount = 2,
         .pStages = shaderStages,
         .pVertexInputState = &vertexInputInfo,
         .pInputAssemblyState = &inputAssembly,
         .pViewportState = &viewportState,
         .pRasterizationState = &rasterizer,
         .pMultisampleState = &multisampling,
         .pDepthStencilState = &depthStencil,
         .pColorBlendState = &colorBlending,
         .pDynamicState = &dynamicState,
         .layout = pipelineLayout,
         .renderPass = nullptr},
        {.colorAttachmentCount = static_cast<uint32_t>(colorFormats.size()),
         .pColorAttachmentFormats = colorFormats.data(),
         .depthAttachmentFormat = findDepthFormat()}};

    backgroundPipeline =
        vk::raii::Pipeline(device, nullptr, pipelineCreateInfoChain.get<vk::GraphicsPipelineCreateInfo>());
}

void VulkanApp::createCommandPool()
{
    vk::CommandPoolCreateInfo poolInfo{.flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
                                       .queueFamilyIndex = queueIndex};
    commandPool = vk::raii::CommandPool(device, poolInfo);
}

void VulkanApp::createCommandBuffers()
{
    vk::CommandBufferAllocateInfo allocInfo{.commandPool = commandPool,
                                            .level = vk::CommandBufferLevel::ePrimary,
                                            .commandBufferCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT)};
    commandBuffers = vk::raii::CommandBuffers(device, allocInfo);
}

void VulkanApp::transition_image_layout(
    uint32_t imageIndex, vk::ImageLayout old_layout,
    vk::ImageLayout new_layout, vk::AccessFlags2 src_access_mask,
    vk::AccessFlags2 dst_access_mask,
    vk::PipelineStageFlags2 src_stage_mask,
    vk::PipelineStageFlags2 dst_stage_mask,
    vk::ImageAspectFlags image_aspect_flags,
    uint32_t mipLevels)
{
    vk::ImageMemoryBarrier2 barrier = {.srcStageMask = src_stage_mask,
                                       .srcAccessMask = src_access_mask,
                                       .dstStageMask = dst_stage_mask,
                                       .dstAccessMask = dst_access_mask,
                                       .oldLayout = old_layout,
                                       .newLayout = new_layout,
                                       .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                       .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                       .image = swapChainImages[imageIndex],
                                       .subresourceRange = {.aspectMask = image_aspect_flags,
                                                            .baseMipLevel = 0,
                                                            .levelCount = mipLevels,
                                                            .baseArrayLayer = 0,
                                                            .layerCount = 1}};
    vk::DependencyInfo dependency_info = {
        .dependencyFlags = {}, .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &barrier};
    commandBuffers[frameIndex].pipelineBarrier2(dependency_info);
}

void VulkanApp::transition_image_layout(const vk::raii::Image *image, vk::ImageLayout old_layout, vk::ImageLayout new_layout, vk::AccessFlags2 src_access_mask, vk::AccessFlags2 dst_access_mask, vk::PipelineStageFlags2 src_stage_mask, vk::PipelineStageFlags2 dst_stage_mask, vk::ImageAspectFlags image_aspect_flags, uint32_t mipLevels)
{
    vk::ImageMemoryBarrier2 barrier = {.srcStageMask = src_stage_mask,
                                       .srcAccessMask = src_access_mask,
                                       .dstStageMask = dst_stage_mask,
                                       .dstAccessMask = dst_access_mask,
                                       .oldLayout = old_layout,
                                       .newLayout = new_layout,
                                       .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                       .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                       .image = *image,
                                       .subresourceRange = {.aspectMask = image_aspect_flags,
                                                            .baseMipLevel = 0,
                                                            .levelCount = mipLevels,
                                                            .baseArrayLayer = 0,
                                                            .layerCount = 1}};
    vk::DependencyInfo dependency_info = {
        .dependencyFlags = {}, .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &barrier};
    commandBuffers[frameIndex].pipelineBarrier2(dependency_info);
}

void transitionImageLayout(vk::raii::CommandBuffer &commandBuffer, const vk::raii::Image &image,
                           vk::ImageLayout oldLayout, vk::ImageLayout newLayout, uint32_t mipLevels)
{
    vk::ImageMemoryBarrier barrier{
        .oldLayout = oldLayout,
        .newLayout = newLayout,
        .srcQueueFamilyIndex = vk::QueueFamilyIgnored,
        .dstQueueFamilyIndex = vk::QueueFamilyIgnored,
        .image = image,
        .subresourceRange = {.aspectMask = vk::ImageAspectFlagBits::eColor, .levelCount = mipLevels, .layerCount = 1}};

    vk::PipelineStageFlags sourceStage;
    vk::PipelineStageFlags destinationStage;

    if (oldLayout == vk::ImageLayout::eUndefined && newLayout == vk::ImageLayout::eTransferDstOptimal)
    {
        barrier.srcAccessMask = vk::AccessFlags{};
        barrier.dstAccessMask = vk::AccessFlagBits::eTransferWrite;

        sourceStage = vk::PipelineStageFlagBits::eTopOfPipe;
        destinationStage = vk::PipelineStageFlagBits::eTransfer;
    }
    else if (oldLayout == vk::ImageLayout::eTransferDstOptimal &&
             newLayout == vk::ImageLayout::eShaderReadOnlyOptimal)
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

void VulkanApp::createSyncObjects()
{
    assert(presentCompleteSemaphores.empty() && renderFinishedSemaphores.empty() && inFlightFences.empty());

    for (size_t i = 0; i < swapChainImages.size(); i++)
    {
        renderFinishedSemaphores.emplace_back(device, vk::SemaphoreCreateInfo());
    }

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
    {
        presentCompleteSemaphores.emplace_back(device, vk::SemaphoreCreateInfo());
        inFlightFences.emplace_back(device, vk::FenceCreateInfo{.flags = vk::FenceCreateFlagBits::eSignaled});
    }
}

void VulkanApp::createTlas()
{
    uint32_t maxInstances = 0;
    uint32_t maxPrimitives = 0;
    for (const auto &model : models)
    {
        maxInstances += model->getMeshInstanceCount();
        maxPrimitives += model->getPrimitiveInstanceCount();
    }
    if (maxInstances == 0)
        return;

    vk::DeviceSize bufferSize = sizeof(vk::AccelerationStructureInstanceKHR) * maxInstances;
    vk::DeviceSize instanceDataBufferSize = sizeof(InstanceData) * maxPrimitives;

    instancesBuffers.clear();
    instancesBuffersMemory.clear();
    instancesBuffersMapped.clear();
    instanceDataBuffers.clear();
    instanceDataBuffersMemory.clear();
    instanceDataBuffersMapped.clear();
    tlasBuffers.clear();
    tlasBuffersMemory.clear();
    tlasHandles.clear();
    tlasScratchBuffers.clear();
    tlasScratchBuffersMemory.clear();

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
    {
        auto [instBuf, instMem] = VulkanUtils::createBuffer(
            device, physicalDevice, bufferSize,
            vk::BufferUsageFlagBits::eShaderDeviceAddress | vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
        instancesBuffers.push_back(std::move(instBuf));
        instancesBuffersMapped.push_back(instMem.mapMemory(0, bufferSize));
        instancesBuffersMemory.push_back(std::move(instMem));

        auto [dataBuf, dataMem] = VulkanUtils::createBuffer(
            device, physicalDevice, instanceDataBufferSize,
            vk::BufferUsageFlagBits::eStorageBuffer,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
        instanceDataBuffers.push_back(std::move(dataBuf));
        instanceDataBuffersMapped.push_back(dataMem.mapMemory(0, instanceDataBufferSize));
        instanceDataBuffersMemory.push_back(std::move(dataMem));

        vk::BufferDeviceAddressInfo instancesAddrInfo{.buffer = *instancesBuffers[i]};
        vk::DeviceAddress instancesDeviceAddress = device.getBufferAddress(instancesAddrInfo);
        vk::AccelerationStructureGeometryInstancesDataKHR instancesData{
            .arrayOfPointers = vk::False,
            .data = instancesDeviceAddress};
        vk::AccelerationStructureGeometryKHR geometry{
            .geometryType = vk::GeometryTypeKHR::eInstances,
            .geometry = instancesData};
        vk::AccelerationStructureBuildGeometryInfoKHR buildInfo{
            .type = vk::AccelerationStructureTypeKHR::eTopLevel,
            .flags = vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastBuild,
            .mode = vk::BuildAccelerationStructureModeKHR::eBuild,
            .geometryCount = 1,
            .pGeometries = &geometry};
        vk::AccelerationStructureBuildSizesInfoKHR buildSizes = device.getAccelerationStructureBuildSizesKHR(
            vk::AccelerationStructureBuildTypeKHR::eDevice,
            buildInfo,
            maxInstances);

        auto [tlBuf, tlMem] = VulkanUtils::createBuffer(
            device, physicalDevice, buildSizes.accelerationStructureSize,
            vk::BufferUsageFlagBits::eAccelerationStructureStorageKHR | vk::BufferUsageFlagBits::eShaderDeviceAddress,
            vk::MemoryPropertyFlagBits::eDeviceLocal);
        tlasBuffers.push_back(std::move(tlBuf));

        vk::AccelerationStructureCreateInfoKHR createInfo{
            .buffer = *tlasBuffers[i],
            .offset = 0,
            .size = buildSizes.accelerationStructureSize,
            .type = vk::AccelerationStructureTypeKHR::eTopLevel};
        tlasHandles.push_back(device.createAccelerationStructureKHR(createInfo));
        tlasBuffersMemory.push_back(std::move(tlMem));

        auto [scBuf, scMem] = VulkanUtils::createBuffer(
            device, physicalDevice, buildSizes.buildScratchSize,
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress,
            vk::MemoryPropertyFlagBits::eDeviceLocal);
        tlasScratchBuffers.push_back(std::move(scBuf));
        tlasScratchBuffersMemory.push_back(std::move(scMem));
    }
}

void VulkanApp::updateTlasInstances(uint32_t currentFrame)
{
    std::vector<vk::AccelerationStructureInstanceKHR> instances;
    std::vector<InstanceData> instanceData;

    uint32_t customIndexOffset = 0;
    for (const auto &model : models)
    {
        model->populateTlasInstances(instances, instanceData, device, customIndexOffset);
    }

    blasInstancesCount = static_cast<uint32_t>(instances.size());

    if (instances.empty() || currentFrame >= instancesBuffersMapped.size())
        return;

    std::memcpy(instancesBuffersMapped[currentFrame], instances.data(), sizeof(vk::AccelerationStructureInstanceKHR) * instances.size());
    if (!instanceData.empty())
    {
        std::memcpy(instanceDataBuffersMapped[currentFrame], instanceData.data(), sizeof(InstanceData) * instanceData.size());
    }
}

void VulkanApp::createUniformBuffers()
{
    cameraUniformBuffers.clear();
    cameraUniformBuffersMemory.clear();
    cameraUniformBuffersMapped.clear();

    vk::DeviceSize bufferSize = sizeof(CameraUBO);

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
    {
        auto [buffer, bufferMem] = VulkanUtils::createBuffer(
            device,
            physicalDevice,
            bufferSize,
            vk::BufferUsageFlagBits::eUniformBuffer,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);

        cameraUniformBuffersMapped.emplace_back(bufferMem.mapMemory(0, bufferSize));

        cameraUniformBuffers.emplace_back(std::move(buffer));
        cameraUniformBuffersMemory.emplace_back(std::move(bufferMem));
    }
}

void VulkanApp::createDepthResources()
{
    vk::Format depthFormat = findDepthFormat();

    std::tie(depthImage, depthImageMemory) = VulkanUtils::createImage(device, physicalDevice, swapChainExtent.width, swapChainExtent.height, depthFormat, vk::ImageTiling::eOptimal, vk::ImageUsageFlagBits::eDepthStencilAttachment | vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eDeviceLocal, msaaSamples);
    depthImageView = VulkanUtils::createImageView(device, *depthImage, depthFormat, vk::ImageAspectFlagBits::eDepth);
}

void VulkanApp::createRenderResources()
{
    vk::Format renderFormat = findSupportedFormat(
        {vk::Format::eR16G16B16A16Sfloat, vk::Format::eR8G8B8A8Unorm, vk::Format::eB8G8R8A8Unorm},
        vk::ImageTiling::eOptimal,
        vk::FormatFeatureFlagBits::eColorAttachment | vk::FormatFeatureFlagBits::eSampledImage);

    vk::ImageUsageFlags usage = vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferSrc;

    for (int i = 0; i < sizeof(renderImages) / sizeof(vk::raii::Image); i++)
    {
        std::tie(renderImages[i], renderImagesMemory[i]) = VulkanUtils::createImage(
            device, physicalDevice, swapChainExtent.width, swapChainExtent.height,
            renderFormat, vk::ImageTiling::eOptimal, usage,
            vk::MemoryPropertyFlagBits::eDeviceLocal, msaaSamples);

        renderImagesView[i] = VulkanUtils::createImageView(
            device, *(renderImages[i]), renderFormat, vk::ImageAspectFlagBits::eColor);
    }
}

void VulkanApp::createCompositionResources()
{
    // Clean up existing resources in the correct RAII order (sets then pool)
    compositionDescriptorSets.clear();
    compositionDescriptorPool = nullptr;
    compositionPipeline = nullptr;
    compositionPipelineLayout = nullptr;
    compositionDescriptorSetLayout = nullptr;

    std::array<vk::DescriptorSetLayoutBinding, 6> bindings = {
        vk::DescriptorSetLayoutBinding{
            .binding = 0,
            .descriptorType = vk::DescriptorType::eSampledImage,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eCompute},
        vk::DescriptorSetLayoutBinding{
            .binding = 1,
            .descriptorType = vk::DescriptorType::eSampledImage,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eCompute},
        vk::DescriptorSetLayoutBinding{
            .binding = 2,
            .descriptorType = vk::DescriptorType::eSampledImage,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eCompute},
        vk::DescriptorSetLayoutBinding{
            .binding = 3,
            .descriptorType = vk::DescriptorType::eSampledImage,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eCompute},
        vk::DescriptorSetLayoutBinding{
            .binding = 4,
            .descriptorType = vk::DescriptorType::eStorageImage,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eCompute},
        vk::DescriptorSetLayoutBinding{
            .binding = 5,
            .descriptorType = vk::DescriptorType::eSampledImage,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eCompute}};

    vk::DescriptorSetLayoutCreateInfo layoutInfo{
        .bindingCount = static_cast<uint32_t>(bindings.size()),
        .pBindings = bindings.data()};
    compositionDescriptorSetLayout = vk::raii::DescriptorSetLayout(device, layoutInfo);

    vk::PushConstantRange pushConstantRange{
        .stageFlags = vk::ShaderStageFlagBits::eCompute,
        .offset = 0,
        .size = sizeof(uint32_t)};

    vk::PipelineLayoutCreateInfo pipelineLayoutInfo{
        .setLayoutCount = 1,
        .pSetLayouts = &(*compositionDescriptorSetLayout),
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &pushConstantRange};
    compositionPipelineLayout = vk::raii::PipelineLayout(device, pipelineLayoutInfo);

    vk::raii::ShaderModule shaderModule = VulkanUtils::createShaderModule(VulkanUtils::readFile("builddir/composition.spv"), device);
    vk::ComputePipelineCreateInfo pipelineInfo{
        .stage = vk::PipelineShaderStageCreateInfo{
            .stage = vk::ShaderStageFlagBits::eCompute,
            .module = *shaderModule,
            .pName = "main"},
        .layout = *compositionPipelineLayout};
    compositionPipeline = vk::raii::Pipeline(device, nullptr, pipelineInfo);

    uint32_t swapChainImageCount = static_cast<uint32_t>(swapChainImageViews.size());
    std::array<vk::DescriptorPoolSize, 2> poolSizes = {
        vk::DescriptorPoolSize{
            .type = vk::DescriptorType::eSampledImage,
            .descriptorCount = swapChainImageCount * 5},
        vk::DescriptorPoolSize{
            .type = vk::DescriptorType::eStorageImage,
            .descriptorCount = swapChainImageCount * 1}};
    vk::DescriptorPoolCreateInfo poolInfo{
        .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
        .maxSets = swapChainImageCount,
        .poolSizeCount = static_cast<uint32_t>(poolSizes.size()),
        .pPoolSizes = poolSizes.data()};
    compositionDescriptorPool = vk::raii::DescriptorPool(device, poolInfo);

    std::vector<vk::DescriptorSetLayout> layouts(swapChainImageCount, *compositionDescriptorSetLayout);
    vk::DescriptorSetAllocateInfo allocInfo{
        .descriptorPool = *compositionDescriptorPool,
        .descriptorSetCount = swapChainImageCount,
        .pSetLayouts = layouts.data()};
    compositionDescriptorSets = vk::raii::DescriptorSets(device, allocInfo);
}

void VulkanApp::createDescriptorPool()
{
    // We need MAX_FRAMES_IN_FLIGHT descriptor sets for both the camera UBO and the TLAS
    std::array poolSize{
        vk::DescriptorPoolSize(vk::DescriptorType::eUniformBuffer, MAX_FRAMES_IN_FLIGHT),
        vk::DescriptorPoolSize(vk::DescriptorType::eAccelerationStructureKHR, MAX_FRAMES_IN_FLIGHT),
        vk::DescriptorPoolSize(vk::DescriptorType::eStorageBuffer, MAX_FRAMES_IN_FLIGHT),
        vk::DescriptorPoolSize(vk::DescriptorType::eCombinedImageSampler, MAX_FRAMES_IN_FLIGHT)};
    vk::DescriptorPoolCreateInfo poolInfo{
        .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
        .maxSets = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT),
        .poolSizeCount = static_cast<uint32_t>(poolSize.size()),
        .pPoolSizes = poolSize.data()};
    descriptorPool = vk::raii::DescriptorPool(device, poolInfo);
}

void VulkanApp::createDescriptorSets()
{
    std::vector<vk::DescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, *descriptorSetLayout);
    vk::DescriptorSetAllocateInfo allocInfo{
        .descriptorPool = *descriptorPool,
        .descriptorSetCount = static_cast<uint32_t>(layouts.size()),
        .pSetLayouts = layouts.data()};

    cameraDescriptorSets.clear();
    cameraDescriptorSets = device.allocateDescriptorSets(allocInfo);

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
    {
        // Binding 0 : La Caméra
        vk::DescriptorBufferInfo bufferInfo{
            .buffer = *cameraUniformBuffers[i],
            .offset = 0,
            .range = sizeof(CameraUBO)};

        // Binding 1 : La TLAS (Acceleration Structure)
        vk::AccelerationStructureKHR rawTlas = *tlasHandles[i];
        vk::WriteDescriptorSetAccelerationStructureKHR asInfo{
            .accelerationStructureCount = 1,
            .pAccelerationStructures = &rawTlas};

        vk::DescriptorBufferInfo instanceDataInfo{
            .buffer = *instanceDataBuffers[i],
            .offset = 0,
            .range = VK_WHOLE_SIZE};

        vk::DescriptorImageInfo imageInfo{
            .sampler = *backgroundTexture.sampler,
            .imageView = *backgroundTexture.imageView,
            .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal
        };

        std::array descriptorWrites{
            vk::WriteDescriptorSet{
                .dstSet = *cameraDescriptorSets[i],
                .dstBinding = 0,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = vk::DescriptorType::eUniformBuffer,
                .pBufferInfo = &bufferInfo},
            vk::WriteDescriptorSet{
                .pNext = &asInfo,
                .dstSet = *cameraDescriptorSets[i],
                .dstBinding = 1,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = vk::DescriptorType::eAccelerationStructureKHR},
            vk::WriteDescriptorSet{
                .dstSet = *cameraDescriptorSets[i],
                .dstBinding = 2,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = vk::DescriptorType::eStorageBuffer,
                .pBufferInfo = &instanceDataInfo},
            vk::WriteDescriptorSet{
                .dstSet = *cameraDescriptorSets[i],
                .dstBinding = 3,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                .pImageInfo = &imageInfo}
            };

        device.updateDescriptorSets(descriptorWrites, {});
    }
}

void VulkanApp::cleanupSwapChain()
{
    swapChainImageViews.clear();
    swapChain = nullptr;
    
    delete ffxMgr;
    ffxMgr = nullptr;
}

void VulkanApp::recreateSwapChain()
{
    int width = 0, height = 0;
    while (width == 0 || height == 0)
    {
        glfwGetFramebufferSize(window, &width, &height);
        glfwWaitEvents();
    }

    device.waitIdle();

    cleanupSwapChain();
    createSwapChain();
    createImageViews();
    createColorResources();
    createDepthResources();
    createRenderResources();
    createCompositionResources();
    initFfx();
}

void VulkanApp::cleanup()
{
    cleanupImGui();

    delete ffxMgr;
    ffxMgr = nullptr;

    physicsEntities.clear();
    models.clear();

    physicsWorld.reset();
    PhysicsWorld::global_shutdown();

    // RAII handles are automatically destroyed in the reverse order of their declaration.
    // Since 'device' is declared before the Vulkan resources, it will safely be destroyed last.

    glfwDestroyWindow(window);
    glfwTerminate();
}

void VulkanApp::generateMipmaps(
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

void VulkanApp::initImGui()
{
    std::vector<vk::DescriptorPoolSize> poolSizes = {
        { vk::DescriptorType::eCombinedImageSampler, 1000 },
        { vk::DescriptorType::eSampledImage, 1000 },
        { vk::DescriptorType::eStorageImage, 1000 },
        { vk::DescriptorType::eUniformBuffer, 1000 },
        { vk::DescriptorType::eStorageBuffer, 1000 }
    };

    vk::DescriptorPoolCreateInfo poolInfo{
        .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
        .maxSets = 1000 * static_cast<uint32_t>(poolSizes.size()),
        .poolSizeCount = static_cast<uint32_t>(poolSizes.size()),
        .pPoolSizes = poolSizes.data()
    };

    imguiDescriptorPool = vk::raii::DescriptorPool(device, poolInfo);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

    ImGui::StyleColorsDark();

    ImGui_ImplGlfw_InitForVulkan(window, true);

    VkFormat colorFormat = static_cast<VkFormat>(swapChainSurfaceFormat.format);

    VkPipelineRenderingCreateInfoKHR pipelineRenderingCreateInfo = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR,
        .colorAttachmentCount = 1,
        .pColorAttachmentFormats = &colorFormat
    };

    ImGui_ImplVulkan_InitInfo init_info = {};
    init_info.Instance = *instance;
    init_info.PhysicalDevice = *physicalDevice;
    init_info.Device = *device;
    init_info.QueueFamily = queueIndex;
    init_info.Queue = *graphicsQueue;
    init_info.PipelineCache = VK_NULL_HANDLE;
    init_info.DescriptorPool = *imguiDescriptorPool;
    init_info.MinImageCount = static_cast<uint32_t>(swapChainImages.size());
    init_info.ImageCount = static_cast<uint32_t>(swapChainImages.size());
    
    // Configuration du Dynamic Rendering sous PipelineInfoMain (ImGui 1.92+)
    init_info.UseDynamicRendering = true;
    init_info.PipelineInfoMain.PipelineRenderingCreateInfo = pipelineRenderingCreateInfo;
    init_info.PipelineInfoMain.MSAASamples = VK_SAMPLE_COUNT_1_BIT;

    ImGui_ImplVulkan_Init(&init_info);
}

void VulkanApp::cleanupImGui()
{
    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    imguiDescriptorPool = nullptr;
}

static void applyCustomTheme(int themeIndex) {
    if (themeIndex == 0) {
        ImGui::StyleColorsDark();
    } else if (themeIndex == 1) {
        ImGui::StyleColorsLight();
    } else if (themeIndex == 2) {
        ImGui::StyleColorsClassic();
    } else if (themeIndex == 3) {
        ImGuiStyle& style = ImGui::GetStyle();
        ImVec4* colors = style.Colors;
        colors[ImGuiCol_Text]                  = ImVec4(1.00f, 1.00f, 1.00f, 1.00f);
        colors[ImGuiCol_TextDisabled]          = ImVec4(0.50f, 0.50f, 0.50f, 1.00f);
        colors[ImGuiCol_WindowBg]              = ImVec4(0.10f, 0.10f, 0.14f, 0.94f);
        colors[ImGuiCol_ChildBg]               = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
        colors[ImGuiCol_PopupBg]               = ImVec4(0.14f, 0.14f, 0.19f, 0.94f);
        colors[ImGuiCol_Border]                = ImVec4(0.30f, 0.25f, 0.45f, 0.60f);
        colors[ImGuiCol_FrameBg]               = ImVec4(0.18f, 0.17f, 0.26f, 0.60f);
        colors[ImGuiCol_FrameBgHovered]        = ImVec4(0.28f, 0.25f, 0.42f, 0.50f);
        colors[ImGuiCol_FrameBgActive]         = ImVec4(0.38f, 0.32f, 0.55f, 0.70f);
        colors[ImGuiCol_TitleBg]               = ImVec4(0.08f, 0.07f, 0.14f, 1.00f);
        colors[ImGuiCol_TitleBgActive]         = ImVec4(0.22f, 0.18f, 0.40f, 1.00f);
        colors[ImGuiCol_TitleBgCollapsed]      = ImVec4(0.00f, 0.00f, 0.00f, 0.50f);
        colors[ImGuiCol_CheckMark]             = ImVec4(0.90f, 0.70f, 0.20f, 1.00f);
        colors[ImGuiCol_SliderGrab]            = ImVec4(0.90f, 0.70f, 0.20f, 1.00f);
        colors[ImGuiCol_SliderGrabActive]      = ImVec4(1.00f, 0.80f, 0.30f, 1.00f);
        colors[ImGuiCol_Button]                = ImVec4(0.24f, 0.20f, 0.40f, 0.85f);
        colors[ImGuiCol_ButtonHovered]         = ImVec4(0.36f, 0.30f, 0.58f, 1.00f);
        colors[ImGuiCol_ButtonActive]          = ImVec4(0.48f, 0.40f, 0.72f, 1.00f);
        colors[ImGuiCol_Header]                = ImVec4(0.24f, 0.20f, 0.40f, 0.75f);
        colors[ImGuiCol_HeaderHovered]         = ImVec4(0.36f, 0.30f, 0.58f, 0.85f);
        colors[ImGuiCol_HeaderActive]          = ImVec4(0.48f, 0.40f, 0.72f, 1.00f);
        colors[ImGuiCol_Tab]                   = ImVec4(0.18f, 0.15f, 0.28f, 0.85f);
        colors[ImGuiCol_TabHovered]            = ImVec4(0.36f, 0.30f, 0.58f, 0.85f);
        colors[ImGuiCol_TabActive]             = ImVec4(0.30f, 0.24f, 0.50f, 1.00f);
    }
}

void VulkanApp::renderUI()
{
    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    // Historique FPS
    fpsHistory[fpsHistoryOffset] = ImGui::GetIO().Framerate;
    fpsHistoryOffset = (fpsHistoryOffset + 1) % 100;

    // Masquer le panneau de paramètres lorsque l'on est en jeu (uiMode == false)
    if (!uiMode) {
        ImGui::Render();
        return;
    }

    // Positionnement et dimensionnement à 3/4 de la fenêtre (75% width & height)
    ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImVec2 workPos = viewport->WorkPos;
    ImVec2 workSize = viewport->WorkSize;

    float winWidth = workSize.x * 0.75f;
    float winHeight = workSize.y * 0.75f;
    float posX = workPos.x + (workSize.x - winWidth) * 0.5f;
    float posY = workPos.y + (workSize.y - winHeight) * 0.5f;

    ImGui::SetNextWindowPos(ImVec2(posX, posY), ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(winWidth, winHeight), ImGuiCond_Always);

    ImGuiWindowFlags windowFlags = ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse;

    if (ImGui::Begin("Paramètres - Moteur Vulkan", nullptr, windowFlags)) {
        // En-tête : Information et bouton pour fermer le menu
        ImGui::TextColored(ImVec4(0.4f, 0.8f, 1.0f, 1.0f), "Panneau de Réglages");
        ImGui::SameLine(ImGui::GetWindowWidth() - 200);
        if (ImGui::Button("Fermer les Réglages (TAB)", ImVec2(180, 0))) {
            uiMode = false;
            glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
        }
        ImGui::Separator();

        if (ImGui::BeginTabBar("ControlTabs", ImGuiTabBarFlags_None)) {

            // TAB 1: GRAPHISMES & SHADERS
            if (ImGui::BeginTabItem("Graphismes")) {
                ImGui::TextColored(ImVec4(0.4f, 0.9f, 1.0f, 1.0f), "--- Caméra & Vision ---");
                if (camera) {
                    ImGui::SliderFloat("Champ de Vision (FOV)", &camera->Zoom, 10.0f, 120.0f, "%.1f°");
                }

                ImGui::Separator();
                ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.2f, 1.0f), "--- Dénoyeur AMD FidelityFX ---");
                if (ffxMgr) {
                    ImGui::SliderInt("Échantillons Max (maxSamples)", &ffxMgr->maxSamples, 1, 64);
                }

                ImGui::Separator();
                ImGui::TextColored(ImVec4(0.3f, 0.9f, 0.5f, 1.0f), "--- Ray-Tracing & Effets ---");
                ImGui::Checkbox("Activer l'Occlusion Ambiante Ray-Tracée (RTAO)", &logicEngine->enableRtao);
                ImGui::Checkbox("Activer les Réflexions Ray-Tracing", &logicEngine->enableReflections);

                ImGui::Separator();
                ImGui::TextColored(ImVec4(0.8f, 0.8f, 0.8f, 1.0f), "--- Paramètres de Résolution & Rendu ---");
                ImGui::Text("Résolution Rendu : %u x %u px", swapChainExtent.width, swapChainExtent.height);

                ImGui::EndTabItem();
            }

            // TAB 2: SOLEIL & CIEL
            if (ImGui::BeginTabItem("Soleil & Ciel")) {
                ImGui::TextColored(ImVec4(1.0f, 0.9f, 0.4f, 1.0f), "--- Simulation du Soleil ---");

                float h = static_cast<float>(logicEngine->hourUTC);
                if (ImGui::SliderFloat("Heure (UTC)", &h, 0.0f, 23.99f, "%.2f h")) {
                    logicEngine->hourUTC = static_cast<double>(h);
                }

                ImGui::Checkbox("Cycle Jour / Nuit automatique", &logicEngine->autoTimeCycle);
                if (logicEngine->autoTimeCycle) {
                    ImGui::SliderFloat("Vitesse du cycle", &logicEngine->timeCycleSpeed, 0.1f, 10.0f, "x%.1f");
                }

                ImGui::Separator();
                ImGui::TextColored(ImVec4(0.8f, 0.8f, 0.8f, 1.0f), "--- Date Calendrière ---");
                ImGui::InputInt("Jour", &logicEngine->day);
                logicEngine->day = std::clamp(logicEngine->day, 1, 31);

                ImGui::InputInt("Mois", &logicEngine->month);
                logicEngine->month = std::clamp(logicEngine->month, 1, 12);

                ImGui::InputInt("Année", &logicEngine->year);
                logicEngine->year = std::clamp(logicEngine->year, 1900, 2100);

                ImGui::Separator();
                ImGui::TextColored(ImVec4(0.8f, 0.8f, 0.8f, 1.0f), "--- Position Géographique ---");
                float lat = static_cast<float>(logicEngine->latitude);
                if (ImGui::SliderFloat("Latitude (°)", &lat, -90.0f, 90.0f, "%.2f°")) {
                    logicEngine->latitude = static_cast<double>(lat);
                }
                float lon = static_cast<float>(logicEngine->longitude);
                if (ImGui::SliderFloat("Longitude (°)", &lon, -180.0f, 180.0f, "%.2f°")) {
                    logicEngine->longitude = static_cast<double>(lon);
                }

                ImGui::Text("Villes prédéfinies :");
                if (ImGui::Button("Paris")) { logicEngine->latitude = 48.8566; logicEngine->longitude = 2.3522; }
                ImGui::SameLine();
                if (ImGui::Button("Tokyo")) { logicEngine->latitude = 35.6762; logicEngine->longitude = 139.6503; }
                ImGui::SameLine();
                if (ImGui::Button("New York")) { logicEngine->latitude = 40.7128; logicEngine->longitude = -74.0060; }
                ImGui::SameLine();
                if (ImGui::Button("Équateur")) { logicEngine->latitude = 0.0; logicEngine->longitude = 0.0; }

                ImGui::Separator();
                glm::vec3 sunDir = SunCalc::calculateSunDirection(
                    logicEngine->year, logicEngine->month, logicEngine->day,
                    logicEngine->hourUTC, logicEngine->latitude, logicEngine->longitude);
                ImGui::Text("Direction du Soleil : (%.2f, %.2f, %.2f)", sunDir.x, sunDir.y, sunDir.z);
                float elevationDeg = std::asin(std::clamp(sunDir.y, -1.0f, 1.0f)) * (180.0f / 3.14159265f);
                ImGui::Text("Élévation du Soleil : %.1f° (%s)", elevationDeg, elevationDeg > 0.0f ? "Jour" : "Nuit");

                ImGui::EndTabItem();
            }

            // TAB 3: PERFORMANCES
            if (ImGui::BeginTabItem("Performances")) {
                ImGui::Text("Taux de rafraîchissement (FPS) : %.1f", ImGui::GetIO().Framerate);
                ImGui::Text("Temps d'affichage (Frametime) : %.3f ms", 1000.0f / ImGui::GetIO().Framerate);
                ImGui::PlotLines("Historique FPS", fpsHistory, 100, fpsHistoryOffset, nullptr, 0.0f, 160.0f, ImVec2(0, 70));
                
                ImGui::Separator();
                ImGui::Text("Résolution Swapchain : %u x %u px", swapChainExtent.width, swapChainExtent.height);
                ImGui::Text("Modèles 3D chargés : %zu", models.size());
                
                ImGui::EndTabItem();
            }

            // TAB 4: THÈME & INTERFACE
            if (ImGui::BeginTabItem("Interface UI")) {
                ImGui::TextColored(ImVec4(0.9f, 0.9f, 0.9f, 1.0f), "--- Style Visuel Dear ImGui ---");
                
                const char* themes[] = { "Sombre (Dark)", "Clair (Light)", "Classique (Classic)", "Moderne Violet/Or" };
                if (ImGui::Combo("Thème de l'UI", &currentThemeIndex, themes, IM_ARRAYSIZE(themes))) {
                    applyCustomTheme(currentThemeIndex);
                }

                ImGui::Separator();
                ImGui::SliderFloat("Opacité de la Fenêtre (Alpha)", &ImGui::GetStyle().Alpha, 0.2f, 1.0f, "%.2f");
                ImGui::SliderFloat("Arrondi des Coins (Rounding)", &ImGui::GetStyle().WindowRounding, 0.0f, 12.0f, "%.1f px");

                ImGui::EndTabItem();
            }

            ImGui::EndTabBar();
        }
    }
    ImGui::End();

    ImGui::Render();
}
