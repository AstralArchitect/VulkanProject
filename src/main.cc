#include "glm/ext/matrix_transform.hpp"
#include "glm/geometric.hpp"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <tiny_gltf.h>
#define TINYGLTF

#include <iostream>

#include "vulkan_app.hpp"

#include "jolt_physics.hpp"
#include <Jolt/Physics/Body/BodyCreationSettings.h>
#include <Jolt/Physics/Collision/Shape/BoxShape.h>

const int MAX_FRAMES_IN_FLIGHT = 2;

char followingmode = 0;

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

    // 1. Mise à jour des matrices des modèles physiques depuis Jolt
    for (auto &entity : physicsEntities)
    {
        glm::mat4 objectMatrix = physicsWorld->get_body_pose(entity.physicsBodyId).to_matrix();
        entity.graphicModel->setModelTransform(objectMatrix);
    }

    for (auto &model : models)
    {
        model->draw(commandBuffer, pipelineLayout);
    }

    commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *backgroundPipeline);
    commandBuffer.draw(3, 1, 0, 0);

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
    vk::DescriptorImageInfo roughnessInfo{
        .imageView = *renderImagesView[2],
        .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal};

    std::array<vk::WriteDescriptorSet, 6> descriptorWrites = {
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
            .pImageInfo = &outputImageInfo},
        vk::WriteDescriptorSet{
            .dstSet = *compositionDescriptorSets[imageIndex],
            .dstBinding = 5,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eSampledImage,
            .pImageInfo = &roughnessInfo}};

    device.updateDescriptorSets(descriptorWrites, nullptr);
}

void VulkanApp::updateUniformBuffer(uint32_t currentImage)
{
    static glm::mat4 previousViewProj = glm::mat4(1.f);

    static glm::vec3 previousCamPoses[3] = {glm::vec3(1.f), glm::vec3(1.f), glm::vec3(1.f)};
    static glm::vec3 previousBallPos = glm::vec3(1.f);

    glm::vec3 ballPos = physicsWorld->get_body_pose(physicsEntities[1].physicsBodyId).position;
    float distance = glm::distance(glm::vec3(0.f), ballPos);
    
    if (distance < 10) followingmode = 0;
    //else if (distance < 100) followingmode = 1;
    else followingmode = 2;

    if (followingmode == 0) {
        camera.Position = glm::vec3(0.0f, 2.f, 0.f);
        previousBallPos = ballPos;
        for (int i = 2; i >= 0; i--) {
            if (i == 0) {
                previousCamPoses[i] = camera.Position;
                break;
            }
            previousCamPoses[i] = previousCamPoses[i - 1];
        }
    }
    else if (followingmode == 1) {
        camera.Position = glm::vec3(0.0f, 2.f, 0.f);
        camera.lookAt(ballPos);

        previousBallPos = ballPos;
        for (int i = 2; i >= 0; i--) {
            if (i == 0) {
                previousCamPoses[i] = camera.Position;
                break;
            }
            previousCamPoses[i] = previousCamPoses[i - 1];
        }
    } else if (followingmode == 2) {
        // Get the camera pos from the ball motion vector
        // Get the normalized ball motion vector
        glm::vec3 mVec = ballPos - previousBallPos;
        mVec = glm::normalize(mVec);
        mVec = ballPos == previousBallPos ? glm::vec3(0.f, 0.f, 0.f) : mVec;
        mVec.y = 0.f;

        // Calculate the camera position
        glm::vec3 camPos;
        camPos = ballPos;
        camPos += glm::vec3(0.f, 1.f, 0.f);
        camPos -= mVec * 3.f;

        for (int i = 0; i < 3; i++) {
            camPos += previousCamPoses[i];
        }
        camPos /= 4;

        camera.Position = camPos;
        camera.lookAt(ballPos);

        previousBallPos = ballPos;
        for (int i = 2; i >= 0; i--) {
            if (i == 0) {
                previousCamPoses[i] = camPos;
                break;
            }
            previousCamPoses[i] = previousCamPoses[i - 1];
        }
    }

    // Camera and projection matrices (shared by all objects)
    glm::mat4 view = camera.GetViewMatrix();
    glm::mat4 proj = glm::perspective(
        glm::radians(45.0f),
        static_cast<float>(swapChainExtent.width) / static_cast<float>(swapChainExtent.height),
        0.1f, 1500.0f);

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

    memcpy(cameraUniformBuffersMapped[currentImage], &ubo, sizeof(ubo));
}

void VulkanApp::drawFrame()
{
    float currentFrame = static_cast<float>(glfwGetTime());
    deltaTime = currentFrame - lastFrame;
    lastFrame = currentFrame;

    // On limite le deltaTime pour éviter que la physique n'explose
    // (notamment à la première frame où le chargement a pris plusieurs secondes !)
    float physicsDeltaTime = deltaTime;
    if (physicsDeltaTime > 0.1f) physicsDeltaTime = 0.1f;

    // input
    // -----
    processInput(window);

    physicsWorld->step(physicsDeltaTime);

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
    camera.Position = glm::vec3(0.f, 2.f, 0.f);
    models.push_back(std::make_unique<GltfModel>(
        "res/models/world.glb",
        device,
        physicalDevice,
        commandPool,
        graphicsQueue,
        textureManager));

    models.back()->setStaticTransform(glm::scale_slow(glm::mat4(1.f), glm::vec3(10.f)));

    JPH::BodyCreationSettings floorSettings(
        models.back()->getMeshShape(), 
        JPH::RVec3(0.0f, -10.0f, 0.0f),
        JPH::Quat::sIdentity(), 
        JPH::EMotionType::Static, 
        Layers::NON_MOVING
    );
    JPH::BodyID floorBodyId = physicsWorld->create_body(floorSettings);

    PhysicsEntity floorEntity;
    floorEntity.physicsBodyId = floorBodyId;
    floorEntity.graphicModel = models.back().get();

    physicsEntities.push_back(floorEntity);

    models.push_back(std::make_unique<GltfModel>(
        "res/models/sphere.glb",
        device,
        physicalDevice,
        commandPool,
        graphicsQueue,
        textureManager));
    
    models.back()->setStaticTransform(glm::scale(glm::mat4(1.f), glm::vec3(.25f)));

    JPH::BodyCreationSettings sphereSettings(
        models.back()->getConvexHull(),
        JPH::RVec3(0.f, 1.f, 0.f),
        JPH::Quat::sIdentity(),
        JPH::EMotionType::Dynamic,
        Layers::MOVING
    );
    sphereSettings.mMotionQuality = JPH::EMotionQuality::LinearCast;
    JPH::BodyID sphereBodyId = physicsWorld->create_body(sphereSettings);

    PhysicsEntity sphereEntity;
    sphereEntity.physicsBodyId = sphereBodyId;
    sphereEntity.graphicModel = models.back().get();

    physicsEntities.push_back(sphereEntity);
}

void VulkanApp::createBackgroundTexture() {
    backgroundTexture = VulkanUtils::loadHDRTexture(device, physicalDevice, commandPool, graphicsQueue, "res/textures/background.hdr");
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