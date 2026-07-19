#include "glm/ext/matrix_float4x4.hpp"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <tiny_gltf.h>
#define TINYGLTF

#include <iostream>

#include <chrono>

#include "vulkan_app.hpp"

const int MAX_FRAMES_IN_FLIGHT = 2;
const double PI = 3.1415926535;

void VulkanApp::recordCommandBuffer(uint32_t imageIndex)
{
    auto &commandBuffer = commandBuffers[frameIndex];

    commandBuffer.begin({});

    vk::BufferDeviceAddressInfo instancesAddrInfo{.buffer = *instancesBuffer};
    vk::DeviceAddress instancesDeviceAddress = device.getBufferAddress(instancesAddrInfo);
    vk::AccelerationStructureGeometryInstancesDataKHR instancesData{
        .arrayOfPointers = vk::False,
        .data = instancesDeviceAddress};
    vk::AccelerationStructureGeometryKHR geometry{
        .geometryType = vk::GeometryTypeKHR::eInstances,
        .geometry = instancesData};

    vk::BufferDeviceAddressInfo scratchAddrInfo{.buffer = *tlasScratchBuffer};
    vk::DeviceAddress scratchAddress = device.getBufferAddress(scratchAddrInfo);

    vk::AccelerationStructureBuildGeometryInfoKHR buildInfo{
        .type = vk::AccelerationStructureTypeKHR::eTopLevel,
        .flags = vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastBuild,
        .mode = vk::BuildAccelerationStructureModeKHR::eBuild,
        .dstAccelerationStructure = *tlasHandle,
        .geometryCount = 1,
        .pGeometries = &geometry,
        .scratchData = scratchAddress};
    vk::AccelerationStructureBuildRangeInfoKHR buildRange{
        .primitiveCount = blasInstancesCount,
        .primitiveOffset = 0,
        .firstVertex = 0,
        .transformOffset = 0};
    const vk::AccelerationStructureBuildRangeInfoKHR *pBuildRange = &buildRange;
    commandBuffer.buildAccelerationStructuresKHR(buildInfo, pBuildRange);
    vk::MemoryBarrier barrier{
        .srcAccessMask = vk::AccessFlagBits::eAccelerationStructureWriteKHR,
        .dstAccessMask = vk::AccessFlagBits::eAccelerationStructureReadKHR};
    commandBuffer.pipelineBarrier(
        vk::PipelineStageFlagBits::eAccelerationStructureBuildKHR,
        vk::PipelineStageFlagBits::eFragmentShader | vk::PipelineStageFlagBits::eComputeShader,
        {}, barrier, nullptr, nullptr);

    // Before starting rendering, transition the swapchain image to
    // vk::ImageLayout::eGeneral (for compute writing)
    transition_image_layout(
        imageIndex, vk::ImageLayout::eUndefined,
        vk::ImageLayout::eGeneral,
        {},                                                 // srcAccessMask
        vk::AccessFlagBits2::eShaderWrite,                  // dstAccessMask
        vk::PipelineStageFlagBits2::eColorAttachmentOutput, // srcStage
        vk::PipelineStageFlagBits2::eComputeShader,         // dstStage
        vk::ImageAspectFlagBits::eColor);

    // Transition depth image to depth attachment optimal layout
    transition_image_layout(
        &depthImage, vk::ImageLayout::eUndefined,
        vk::ImageLayout::eDepthAttachmentOptimal,
        vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
        vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
        vk::PipelineStageFlagBits2::eEarlyFragmentTests |
            vk::PipelineStageFlagBits2::eLateFragmentTests,
        vk::PipelineStageFlagBits2::eEarlyFragmentTests |
            vk::PipelineStageFlagBits2::eLateFragmentTests,
        vk::ImageAspectFlagBits::eDepth);

    // Transition color image to color attachment optimal layout
    transition_image_layout(
        &colorImage, vk::ImageLayout::eUndefined,
        vk::ImageLayout::eColorAttachmentOptimal,
        {},                                                 // srcAccessMask
        vk::AccessFlagBits2::eColorAttachmentWrite,         // dstAccessMask
        vk::PipelineStageFlagBits2::eColorAttachmentOutput, // srcStage
        vk::PipelineStageFlagBits2::eColorAttachmentOutput, // dstStage
        vk::ImageAspectFlagBits::eColor);

    for (int i = 0; i < sizeof(renderImages) / sizeof(vk::raii::Image); i++) {
    transition_image_layout(
        &(renderImages[i]), vk::ImageLayout::eUndefined,
        vk::ImageLayout::eColorAttachmentOptimal,
        {},                                                 // srcAccessMask
        vk::AccessFlagBits2::eColorAttachmentWrite,         // dstAccessMask
        vk::PipelineStageFlagBits2::eColorAttachmentOutput, // srcStage
        vk::PipelineStageFlagBits2::eColorAttachmentOutput, // dstStage
        vk::ImageAspectFlagBits::eColor);
    }

    vk::ClearValue clearColor = vk::ClearColorValue(0.0f, 0.0f, 0.0f, 1.f);
    vk::ClearValue clearDepth = vk::ClearDepthStencilValue(1.0f, 0);

    vk::RenderingAttachmentInfo depthAttachmentInfo = {
        .imageView = depthImageView,
        .imageLayout = vk::ImageLayout::eDepthAttachmentOptimal,
        .loadOp = vk::AttachmentLoadOp::eClear,
        .storeOp = vk::AttachmentStoreOp::eDontCare,
        .clearValue = clearDepth};

    std::array<vk::RenderingAttachmentInfo, 6> colorAttachments = {};
    colorAttachments[0] = vk::RenderingAttachmentInfo{
        .imageView = *colorImageView,
        .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
        .loadOp = vk::AttachmentLoadOp::eClear,
        .storeOp = vk::AttachmentStoreOp::eStore,
        .clearValue = clearColor
    };
    colorAttachments[1] = vk::RenderingAttachmentInfo{
        .imageView = *renderImagesView[0],
        .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
        .loadOp = vk::AttachmentLoadOp::eClear,
        .storeOp = vk::AttachmentStoreOp::eStore,
        .clearValue = vk::ClearColorValue(0.0f, 0.0f, 0.0f, 0.0f)
    };
    colorAttachments[2] = vk::RenderingAttachmentInfo{
        .imageView = *renderImagesView[1],
        .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
        .loadOp = vk::AttachmentLoadOp::eClear,
        .storeOp = vk::AttachmentStoreOp::eStore,
        .clearValue = vk::ClearColorValue(0.0f, 0.0f, 0.0f, 0.0f)
    };
    colorAttachments[3] = vk::RenderingAttachmentInfo{
        .imageView = *renderImagesView[2],
        .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
        .loadOp = vk::AttachmentLoadOp::eClear,
        .storeOp = vk::AttachmentStoreOp::eStore,
        .clearValue = vk::ClearColorValue(0.0f, 0.0f, 0.0f, 0.0f)
    };
    colorAttachments[4] = vk::RenderingAttachmentInfo{
        .imageView = *renderImagesView[3],
        .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
        .loadOp = vk::AttachmentLoadOp::eClear,
        .storeOp = vk::AttachmentStoreOp::eStore,
        .clearValue = vk::ClearColorValue(0.0f, 0.0f, 0.0f, 0.0f)
    };
    colorAttachments[5] = vk::RenderingAttachmentInfo{
        .imageView = *renderImagesView[4],
        .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
        .loadOp = vk::AttachmentLoadOp::eClear,
        .storeOp = vk::AttachmentStoreOp::eStore,
        .clearValue = vk::ClearColorValue(0.0f, 0.0f, 0.0f, 0.0f)
    };
    vk::RenderingInfo renderingInfo = {
        .renderArea = {.offset = {0, 0}, .extent = swapChainExtent},
        .layerCount = 1,
        .colorAttachmentCount = static_cast<uint32_t>(colorAttachments.size()),
        .pColorAttachments = colorAttachments.data(),
        .pDepthAttachment = &depthAttachmentInfo
    };

    commandBuffer.beginRendering(renderingInfo);

    commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *graphicsPipeline);

    commandBuffer.setViewport(0, vk::Viewport(0.0f, 0.0f, static_cast<float>(swapChainExtent.width), static_cast<float>(swapChainExtent.height), 0.0f, 1.0f));
    commandBuffer.setScissor(0, vk::Rect2D(vk::Offset2D(0, 0), swapChainExtent));

    // Liaison des descripteurs pour la caméra/uniforms
    commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *pipelineLayout, 0, *cameraDescriptorSets[frameIndex], nullptr);

    commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *pipelineLayout, 1, *textureManager.getDescriptorSet(), nullptr);

    for (auto &model : models)
    {
        model->draw(commandBuffer, pipelineLayout);
    }

    commandBuffer.endRendering();

    transition_image_layout(
        &colorImage, vk::ImageLayout::eColorAttachmentOptimal,
        vk::ImageLayout::eShaderReadOnlyOptimal,
        vk::AccessFlagBits2::eColorAttachmentWrite,
        vk::AccessFlagBits2::eShaderRead,
        vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        vk::PipelineStageFlagBits2::eComputeShader,
        vk::ImageAspectFlagBits::eColor);
    for (int i = 0; i < sizeof(renderImages) / sizeof(vk::raii::Image); i++) {
        transition_image_layout(
            &renderImages[i], vk::ImageLayout::eColorAttachmentOptimal,
            vk::ImageLayout::eShaderReadOnlyOptimal,
            vk::AccessFlagBits2::eColorAttachmentWrite,
            vk::AccessFlagBits2::eShaderRead,
            vk::PipelineStageFlagBits2::eColorAttachmentOutput,
            vk::PipelineStageFlagBits2::eComputeShader,
            vk::ImageAspectFlagBits::eColor);
    }
    transition_image_layout(
        &depthImage, vk::ImageLayout::eDepthAttachmentOptimal,
        vk::ImageLayout::eShaderReadOnlyOptimal,
        vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
        vk::AccessFlagBits2::eShaderRead,
        vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eLateFragmentTests,
        vk::PipelineStageFlagBits2::eComputeShader,
        vk::ImageAspectFlagBits::eDepth);

    ffxMgr->dispatchDenoiser(commandBuffer, frameIndex, swapChainExtent.width, swapChainExtent.height, *depthImage, *renderImages[1], *renderImages[2]);

    commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, *compositionPipeline);
    
    updateCompositionDescriptorSet(imageIndex, frameIndex);

    commandBuffer.bindDescriptorSets(
        vk::PipelineBindPoint::eCompute,
        *compositionPipelineLayout,
        0,
        *compositionDescriptorSets[imageIndex],
        nullptr);
    uint32_t groupCountX = (swapChainExtent.width + 15) / 16;
    uint32_t groupCountY = (swapChainExtent.height + 15) / 16;
    commandBuffer.dispatch(groupCountX, groupCountY, 1);

    transition_image_layout(
        imageIndex, vk::ImageLayout::eGeneral,
        vk::ImageLayout::ePresentSrcKHR,
        vk::AccessFlagBits2::eShaderWrite,
        {},
        vk::PipelineStageFlagBits2::eComputeShader,
        vk::PipelineStageFlagBits2::eBottomOfPipe,
        vk::ImageAspectFlagBits::eColor);

    commandBuffer.end();
}

void VulkanApp::updateCompositionDescriptorSet(uint32_t imageIndex, uint32_t frameIndex)
{
    vk::DescriptorImageInfo directColorInfo{
        .imageView = *colorImageView,
        .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal};
    vk::DescriptorImageInfo reflectionInfo{
        .imageView = *ffxMgr->images.ffxHistoryRadianceViews[((frameIndex % 2) + 1) % 2],
        .imageLayout = vk::ImageLayout::eGeneral};
    vk::DescriptorImageInfo normalInfo{
        .imageView = *renderImagesView[1],
        .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal};
    vk::DescriptorImageInfo depthInfo{
        .imageView = *depthImageView,
        .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal};
    vk::DescriptorImageInfo outputImageInfo{
        .imageView = *swapChainImageViews[imageIndex],
        .imageLayout = vk::ImageLayout::eGeneral};

    std::array<vk::WriteDescriptorSet, 5> descriptorWrites = {
        vk::WriteDescriptorSet{
            .dstSet = *compositionDescriptorSets[imageIndex],
            .dstBinding = 0,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eSampledImage,
            .pImageInfo = &directColorInfo},
        vk::WriteDescriptorSet{
            .dstSet = *compositionDescriptorSets[imageIndex],
            .dstBinding = 1,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eSampledImage,
            .pImageInfo = &reflectionInfo},
        vk::WriteDescriptorSet{
            .dstSet = *compositionDescriptorSets[imageIndex],
            .dstBinding = 2,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eSampledImage,
            .pImageInfo = &normalInfo},
        vk::WriteDescriptorSet{
            .dstSet = *compositionDescriptorSets[imageIndex],
            .dstBinding = 3,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eSampledImage,
            .pImageInfo = &depthInfo},
        vk::WriteDescriptorSet{
            .dstSet = *compositionDescriptorSets[imageIndex],
            .dstBinding = 4,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eStorageImage,
            .pImageInfo = &outputImageInfo}};

    device.updateDescriptorSets(descriptorWrites, nullptr);
}

void VulkanApp::updateUniformBuffer(uint32_t currentImage)
{
    static glm::mat4 previousViewProj = glm::mat4(1.f);

    // Camera and projection matrices (shared by all objects)
    glm::mat4 view = camera.GetViewMatrix();
    glm::mat4 proj = glm::perspective(
        glm::radians(45.0f),
        static_cast<float>(swapChainExtent.width) / static_cast<float>(swapChainExtent.height),
        0.1f, 20.0f);

    proj[1][1] *= -1; // Flip Y for Vulkan

    float time = glfwGetTime();

    CameraUBO ubo{};
    ubo.view = view;
    ubo.prevViewProj = previousViewProj;
    ubo.proj = proj;
    ubo.camPos = glm::vec4(camera.Position, 1.f);
    ubo.time = time;

    ffxMgr->updateConstantsBuffer(ubo.view, ubo.proj, previousViewProj);

    previousViewProj = ubo.proj * ubo.view;

    ubo.lightsPos[0] = glm::vec4(1.5f, 2.f, 0.f, 1.f);
    ubo.lightsPos[1] = glm::vec4(-3.f, 2.f, 1.f, 1.f);

    // Placement des aiguilles
    models[3]->modelTransform = glm::scale(glm::mat4(1.f), glm::vec3(.075f));
    models[3]->modelTransform = glm::translate(models[3]->modelTransform, glm::vec3(0.f, 7.0f, 2.f));
    models[3]->modelTransform = glm::rotate(models[3]->modelTransform, glm::radians(90.f), glm::vec3(0.f, 1.0f, 0.f));
    models[3]->modelTransform = glm::rotate(models[3]->modelTransform, glm::radians(90.f), glm::vec3(0.f, 0.0f, 1.f));
    models[3]->modelTransform = glm::rotate(models[3]->modelTransform, glm::radians(-90.f), glm::vec3(0.f, 1.0f, 0.f));

    models[4]->modelTransform = glm::scale(glm::mat4(1.f), glm::vec3(.075f));
    models[4]->modelTransform = glm::translate(models[4]->modelTransform, glm::vec3(0.f, 7.0f, 2.f));
    models[4]->modelTransform = glm::rotate(models[4]->modelTransform, glm::radians(90.f), glm::vec3(0.f, 1.0f, 0.f));
    models[4]->modelTransform = glm::rotate(models[4]->modelTransform, glm::radians(90.f), glm::vec3(0.f, 0.0f, 1.f));
    models[4]->modelTransform = glm::rotate(models[4]->modelTransform, glm::radians(-90.f), glm::vec3(0.f, 1.0f, 0.f));

    // Alignement sur l'heure actuelle
    // get the actual time
    auto now_time = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());

    // Convert to local time components
    struct tm* local_time = localtime(&now_time);  // or gmtime() for UTC

    int8_t minutes = local_time->tm_min;
    int32_t hours = local_time->tm_hour;

    if (hours == 0) {
        hours = 12; // Convert 0 to 12 for midnight/noon
    }
    // convert the actual time into degrees
    models[3]->modelTransform = glm::rotate(models[3]->modelTransform, glm::radians(minutes * 6.f), glm::vec3(0.0, -1.0, 0.0));
    models[4]->modelTransform = glm::rotate(models[4]->modelTransform, glm::radians(hours * 30.f), glm::vec3(0.0, -1.0, 0.0));

    memcpy(cameraUniformBuffersMapped[currentImage], &ubo, sizeof(ubo));
}

void VulkanApp::drawFrame()
{
    float currentFrame = static_cast<float>(glfwGetTime());
    deltaTime = currentFrame - lastFrame;
    lastFrame = currentFrame;

    // input
    // -----
    processInput(window);

    // Note: inFlightFences, presentCompleteSemaphores, and commandBuffers are
    // indexed by frameIndex,
    //       while renderFinishedSemaphores is indexed by imageIndex
    auto fenceResult =
        device.waitForFences(*inFlightFences[frameIndex], vk::True, UINT64_MAX);
    if (fenceResult != vk::Result::eSuccess)
    {
        throw std::runtime_error("failed to wait for fence!");
    }
    device.resetFences(*inFlightFences[frameIndex]);

    auto [result, imageIndex] = swapChain.acquireNextImage(
        UINT64_MAX, *presentCompleteSemaphores[frameIndex], nullptr);

    if (result == vk::Result::eErrorOutOfDateKHR)
    {
        recreateSwapChain();
        return;
    }
    if (result != vk::Result::eSuccess && result != vk::Result::eSuboptimalKHR)
    {
        assert(result == vk::Result::eTimeout || result == vk::Result::eNotReady);
        throw std::runtime_error("failed to acquire swap chain image!");
    }

    updateUniformBuffer(frameIndex);
    updateTlasInstances();

    commandBuffers[frameIndex].reset();
    recordCommandBuffer(imageIndex);

    vk::PipelineStageFlags waitDestinationStageMask(
        vk::PipelineStageFlagBits::eColorAttachmentOutput);
    const vk::SubmitInfo submitInfo{
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &*presentCompleteSemaphores[frameIndex],
        .pWaitDstStageMask = &waitDestinationStageMask,
        .commandBufferCount = 1,
        .pCommandBuffers = &*commandBuffers[frameIndex],
        .signalSemaphoreCount = 1,
        .pSignalSemaphores = &*renderFinishedSemaphores[imageIndex]};
    graphicsQueue.submit(submitInfo, *inFlightFences[frameIndex]);

    const vk::PresentInfoKHR presentInfoKHR{
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &*renderFinishedSemaphores[imageIndex],
        .swapchainCount = 1,
        .pSwapchains = &*swapChain,
        .pImageIndices = &imageIndex};
    result = graphicsQueue.presentKHR(presentInfoKHR);
    if ((result == vk::Result::eSuboptimalKHR) ||
        (result == vk::Result::eErrorOutOfDateKHR))
    {
        framebufferResized = false;
        recreateSwapChain();
    }
    else
    {
        // There are no other success codes than eSuccess; on any error code,
        // presentKHR already threw an exception.
        assert(result == vk::Result::eSuccess);
    }
    frameIndex = (frameIndex + 1) % MAX_FRAMES_IN_FLIGHT;
}

void VulkanApp::loadModels()
{
    models.push_back(std::make_unique<GltfModel>(
        "res/models/horloge.glb",
        device,
        physicalDevice,
        commandPool,
        graphicsQueue,
        textureManager));

    models[0]->modelTransform = glm::scale(glm::mat4(1.f), glm::vec3(.5f));

    models.push_back(std::make_unique<GltfModel>(
        "res/models/mirror.glb",
        device,
        physicalDevice,
        commandPool,
        graphicsQueue,
        textureManager));

    models[1]->modelTransform = glm::scale(glm::mat4(1.f), glm::vec3(.5f));
    models[1]->modelTransform = glm::translate(models[1]->modelTransform, glm::vec3(-2.f, 0.0f, -1.25f));
    models[1]->modelTransform = glm::rotate(models[1]->modelTransform, glm::radians(225.f), glm::vec3(0.f, 1.0f, 0.f));

    models.push_back(std::make_unique<GltfModel>(
        "res/models/armoire.glb",
        device,
        physicalDevice,
        commandPool,
        graphicsQueue,
        textureManager));

    models[2]->modelTransform = glm::scale(glm::mat4(1.f), glm::vec3(2.f));
    models[2]->modelTransform = glm::translate(models[2]->modelTransform, glm::vec3(0.f, 0.f, 0.f));

    models.push_back(std::make_unique<GltfModel>(
        "res/models/aiguille.glb",
        device,
        physicalDevice,
        commandPool,
        graphicsQueue,
        textureManager));

    models.push_back(std::make_unique<GltfModel>(
        "res/models/aiguille_heure.glb",
        device,
        physicalDevice,
        commandPool,
        graphicsQueue,
        textureManager));

    models.push_back(std::make_unique<GltfModel>(
        "res/models/lampe.glb",
        device,
        physicalDevice,
        commandPool,
        graphicsQueue,
        textureManager));

    models[5]->modelTransform = glm::scale(glm::mat4(1.f), glm::vec3(.5f)) * glm::translate(glm::mat4(1.f), glm::vec3(3.f, 0.f, 0.f));
}

void VulkanApp::mainLoop()
{
    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();

        // Pause si la fenêtre est minimisée
        int width = 0, height = 0;
        glfwGetFramebufferSize(window, &width, &height);
        while (width == 0 || height == 0)
        {
            glfwGetFramebufferSize(window, &width, &height);
            glfwWaitEvents();
        }

        drawFrame();
    }
    device.waitIdle();
}

int main()
{
    try
    {
        VulkanApp app;
        app.init();
        app.run();
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}