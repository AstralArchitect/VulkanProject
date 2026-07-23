#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

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

void VulkanApp::init()
{
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
    static glm::vec3 previousBallPos = glm::vec3(1.f);
    glm::vec3 ballPos = physicsWorld->get_body_pose(physicsEntities[1].physicsBodyId).position;

    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);

    if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS)
    {
        glm::vec3 mVec = ballPos - previousBallPos;
        mVec = glm::normalize(mVec);
        mVec = ballPos == previousBallPos ? glm::vec3(0.f, 0.f, 0.f) : mVec;
        mVec.y = 0.f;
        
        mVec = glm::mat3(glm::rotate(glm::mat4(1.f), glm::radians(90.f), glm::vec3(0.f, 1.f, 0.f))) * mVec;

        mVec *= 200;

        physicsWorld->add_force(physicsEntities[1].physicsBodyId, mVec);
    }
    else if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS)
    {
        glm::vec3 mVec = ballPos - previousBallPos;
        mVec = glm::normalize(mVec);
        mVec = ballPos == previousBallPos ? glm::vec3(0.f, 0.f, 0.f) : mVec;
        mVec.y = 0.f;

        mVec = glm::mat3(glm::rotate(glm::mat4(1.f), glm::radians(90.f), glm::vec3(0.f, -1.f, 0.f))) * mVec;

        mVec *= 200;

        physicsWorld->add_force(physicsEntities[1].physicsBodyId, mVec);
    }
    if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS)
    {
        glm::vec3 mVec = ballPos - previousBallPos;
        mVec = glm::normalize(mVec);
        mVec = ballPos == previousBallPos ? glm::vec3(1.f, 0.f, 0.f) : mVec;
        mVec.y = 0.f;

        mVec *= 200;

        physicsWorld->add_force(physicsEntities[1].physicsBodyId, mVec);
    }
    else if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS)
    {
        glm::vec3 mVec = ballPos - previousBallPos;
        mVec = glm::normalize(mVec);
        mVec = ballPos == previousBallPos ? glm::vec3(0.f, 0.f, 0.f) : mVec;
        mVec.y = 0.f;
        
        mVec *= 200;

        physicsWorld->add_force(physicsEntities[1].physicsBodyId, -mVec);
    }
    else if (glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS)
    {
        PhysicsPose initialPose;
        initialPose.position = glm::vec3(0.0f, 1.0f, 0.0f);
        initialPose.orientation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
        physicsWorld->move_kinematic(physicsEntities[1].physicsBodyId, initialPose);
        physicsWorld->set_linear_velocity(physicsEntities[1].physicsBodyId, glm::vec3(0.0f));

        ballPos = initialPose.position;
    }
    previousBallPos = ballPos;

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
}

// Mouse callback (Logique FPV / First Person View)
void VulkanApp::mouse(double xposIn, double yposIn) {
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

    camera.ProcessMouseMovement(xoffset, yoffset);
}


void VulkanApp::initWindow()
{
    if (!glfwInit())
    {
        throw std::runtime_error("Échec de l'initialisation de GLFW");
    }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

    window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
    glfwSetWindowUserPointer(window, this);
    glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
    glfwSetCursorPosCallback(window, mouseCallback);
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    if (!window)
    {
        throw std::runtime_error("Échec de la création de la fenêtre GLFW");
    }

    camera = Camera(
        glm::vec3(2.0f, 2.0f, 6.0f), // Position
        glm::vec3(0.0f, 1.0f, 0.0f)  // World Up
    );

    camera.lookAt(glm::vec3(0.f, 1.f, 0.f));
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

    for (const auto &device : physicalDevices)
    {
        if (isDeviceSuitable(device, requiredDeviceExtension))
        {
            physicalDevice = device;
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
        vk::DescriptorSetLayoutBinding(0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, nullptr),
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

    std::tie(instancesBuffer, instancesBufferMemory) = VulkanUtils::createBuffer(
        device, physicalDevice, bufferSize,
        vk::BufferUsageFlagBits::eShaderDeviceAddress | vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);

    vk::DeviceSize instanceDataBufferSize = sizeof(InstanceData) * maxPrimitives;
    std::tie(instanceDataBuffer, instanceDataBufferMemory) = VulkanUtils::createBuffer(
        device, physicalDevice, instanceDataBufferSize,
        vk::BufferUsageFlagBits::eStorageBuffer,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    instanceDataBufferMapped = instanceDataBufferMemory.mapMemory(0, instanceDataBufferSize);

    instancesBufferMapped = instancesBufferMemory.mapMemory(0, bufferSize);
    vk::BufferDeviceAddressInfo instancesAddrInfo{.buffer = *instancesBuffer};
    vk::DeviceAddress instancesDeviceAddress = device.getBufferAddress(instancesAddrInfo);
    vk::AccelerationStructureGeometryInstancesDataKHR instancesData{
        .arrayOfPointers = vk::False,
        .data = instancesDeviceAddress};
    vk::AccelerationStructureGeometryKHR geometry{
        .geometryType = vk::GeometryTypeKHR::eInstances,
        .geometry = instancesData};
    vk::AccelerationStructureBuildGeometryInfoKHR buildInfo{
        .type = vk::AccelerationStructureTypeKHR::eTopLevel,
        .flags = vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastBuild, // Rapidité de build privilégiée car reconstruite chaque frame
        .mode = vk::BuildAccelerationStructureModeKHR::eBuild,
        .geometryCount = 1,
        .pGeometries = &geometry};
    vk::AccelerationStructureBuildSizesInfoKHR buildSizes = device.getAccelerationStructureBuildSizesKHR(
        vk::AccelerationStructureBuildTypeKHR::eDevice,
        buildInfo,
        maxInstances);
    std::tie(tlasBuffer, tlasBufferMemory) = VulkanUtils::createBuffer(
        device, physicalDevice, buildSizes.accelerationStructureSize,
        vk::BufferUsageFlagBits::eAccelerationStructureStorageKHR | vk::BufferUsageFlagBits::eShaderDeviceAddress,
        vk::MemoryPropertyFlagBits::eDeviceLocal);

    vk::AccelerationStructureCreateInfoKHR createInfo{
        .buffer = *tlasBuffer,
        .offset = 0,
        .size = buildSizes.accelerationStructureSize,
        .type = vk::AccelerationStructureTypeKHR::eTopLevel};
    tlasHandle = device.createAccelerationStructureKHR(createInfo);

    std::tie(tlasScratchBuffer, tlasScratchBufferMemory) = VulkanUtils::createBuffer(
        device, physicalDevice, buildSizes.buildScratchSize,
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress,
        vk::MemoryPropertyFlagBits::eDeviceLocal);
}

void VulkanApp::updateTlasInstances()
{
    std::vector<vk::AccelerationStructureInstanceKHR> instances;
    std::vector<InstanceData> instanceData;

    uint32_t customIndexOffset = 0;
    for (const auto &model : models)
    {
        model->populateTlasInstances(instances, instanceData, device, customIndexOffset);
    }

    blasInstancesCount = instances.size();

    if (instances.empty())
        return;

    std::memcpy(instancesBufferMapped, instances.data(), sizeof(vk::AccelerationStructureInstanceKHR) * instances.size());
    if (!instanceData.empty())
    {
        std::memcpy(instanceDataBufferMapped, instanceData.data(), sizeof(InstanceData) * instanceData.size());
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

    vk::PipelineLayoutCreateInfo pipelineLayoutInfo{
        .setLayoutCount = 1,
        .pSetLayouts = &(*compositionDescriptorSetLayout)};
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
        vk::AccelerationStructureKHR rawTlas = *tlasHandle;
        vk::WriteDescriptorSetAccelerationStructureKHR asInfo{
            .accelerationStructureCount = 1,
            .pAccelerationStructures = &rawTlas};

        vk::DescriptorBufferInfo instanceDataInfo{
            .buffer = *instanceDataBuffer,
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
    delete ffxMgr;
    ffxMgr = nullptr;

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