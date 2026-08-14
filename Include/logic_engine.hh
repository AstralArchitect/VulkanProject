#pragma once

#include "level_loader.hh"
#include "vulkan_app.hpp"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/hash.hpp>

#include "model.hpp"

enum class PlayerAnimation : int {
    BindPose  = -1,
    Landing   = 0,
    Running   = 1,
    Sprinting = 2,
    Jumping   = 3
};

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

    void updatePlayerMovement(GLFWwindow* window, PhysicsWorld* physicsWorld, float deltaTime);

    void loadModels(VulkanApp* app);

    void nextFrame(const VulkanApp* app); // called by VulkanApp::updateUniformBuffer(uint32_t currentImage);

    Camera* getCamPtr() { return &camera; }

    // Skybox & Sun parameters : defaulted to equinoxe / Paris
    int year = 2026;
    int month = 9;
    int day = 23;
    double hourUTC = 8.0;
    double latitude = 48.8566;
    double longitude = 2.3522;
    bool autoTimeCycle = false;
    float timeCycleSpeed = 1.0f;

    LevelData levelData;

    bool enableRtao;
    bool enableReflections;

    int playerIndex = ~0;
private:
    Camera camera;

    bool wasGrounded = true;
    PlayerAnimation currentAnim = PlayerAnimation::BindPose;
    float animStartTime = 0.0f;

    float lastJumpTime = 0.f;
};