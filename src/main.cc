#include "GLFW/glfw3.h"
#include "model.hpp"
#include "vulkan_utils.hpp"
#include <stdexcept>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <tiny_gltf.h>
#define TINYGLTF

#include <iostream>

#include "vulkan_app.hpp"

#include "jolt_physics.hpp"
#include <Jolt/Physics/Body/BodyCreationSettings.h>
#include <Jolt/Physics/Collision/Shape/BoxShape.h>

#include "logic_engine.hh"

const int MAX_FRAMES_IN_FLIGHT = 2;

char followingmode = 0;

std::vector<std::unique_ptr<GltfModel>> models;
std::vector<PhysicsEntity> physicsEntities;

void VulkanApp::loadModels()
{
    if (logicEngine) {
        logicEngine->loadModels(this);
    } else {
        throw std::runtime_error("logic Engine not initialized !");
    }
}

void VulkanApp::createBackgroundTexture() {
    backgroundTexture.create(device, physicalDevice, commandPool, graphicsQueue, descriptorSetLayout, 1024);
}

void VulkanApp::mainLoop()
{
    lastFrame = static_cast<float>(glfwGetTime());
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

int main() {
    try
    {
        LogicEngine *logicEngine = new LogicEngine();

        VulkanApp app;
        app.setCamera(logicEngine->getCamPtr());
        app.init(logicEngine);
        app.run();

        delete logicEngine;
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}