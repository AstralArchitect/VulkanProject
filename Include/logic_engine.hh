#pragma once

#include "vulkan_app.hpp"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/hash.hpp>

#include "model.hpp"

enum GameMovment {
    M_FORWARD,
    M_BACKWARD,
    M_LEFT,
    M_RIGHT,
    M_JUMP
};

extern std::vector<std::unique_ptr<GltfModel>> models;
extern std::vector<PhysicsEntity> physicsEntities;

class LogicEngine {
public:
    LogicEngine();

    void movment(GameMovment, float deltaTime);

    void loadModels(VulkanApp* app);

    void nextFrame(const VulkanApp* app); // called by VulkanApp::updateUniformBuffer(uint32_t currentImage);

    Camera* getCamPtr() { return &camera; }

    // Skybox parameters : defaulted to equinoxe
    int year = 2026;
    int month = 9;
    int day = 23;
    double hourUTC = 08.0;
private:
    Camera camera;

    bool wasGrounded = true;
    int currentAnimIndex = -1; // -1: bind pose / idle, 0: landing, 1: running, 2: jumping/air
    float animStartTime = 0.0f;

    float lastJumpTime = 0.f;
};