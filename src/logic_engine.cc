#include "logic_engine.hh"

#include "glm/ext/matrix_transform.hpp"
#include "jolt_physics.hpp"
#include <Jolt/Physics/Body/BodyCreationSettings.h>
#include <Jolt/Physics/Collision/Shape/BoxShape.h>
#include <stdexcept>

#include "level_loader.hh"
#include "model.hpp"
#include "physics_world.hpp"
#include "skin.hh"
#include "text_manager.hpp"
#include "vulkan/vulkan_raii.hpp"
#include "vulkan_app.hpp"

LogicEngine::LogicEngine() {
    camera = Camera(
        glm::vec3(2.0f, 2.0f, 6.0f), // Position
        glm::vec3(0.0f, 1.0f, 0.0f)  // World Up
    );

    camera.lookAt(glm::vec3(0.f, 1.f, 0.f));
}

void LogicEngine::movment(GameMovment movement, float deltaTime) {

}

void LogicEngine::updatePlayerMovement(GLFWwindow* window, PhysicsWorld* physicsWorld, float deltaTime) {
    if (physicsEntities.size() <= 1 || !physicsWorld) return;

    PhysicsPose pose = physicsWorld->get_body_pose(physicsEntities[1].physicsBodyId);
    float currentTime = static_cast<float>(glfwGetTime());

    // Détection dynamique du sol par Raycast vers le bas (ignore le corps du joueur)
    glm::vec3 downRayOrigin = pose.position + glm::vec3(0.0f, 0.5f, 0.0f);
    glm::vec3 downRayDir = glm::vec3(0.0f, -1.0f, 0.0f);
    float hitDist = 0.0f;
    glm::vec3 hitNormal(0.0f);
    JPH::BodyID hitBody;

    bool isGrounded = false;
    if (physicsWorld->raycast(downRayOrigin, downRayDir, 2.0f, hitDist, hitNormal, hitBody, physicsEntities[1].physicsBodyId)) {
        if (hitDist <= 1.35f && hitNormal.y >= 0.5f) {
            isGrounded = true;
        }
    }

    if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS)
    {
        pose.orientation = glm::normalize(glm::rotate(pose.orientation, glm::radians(3.0f), glm::vec3(0.f, 1.f, 0.f)));
        physicsWorld->move_kinematic(physicsEntities[1].physicsBodyId, pose);
    }
    else if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS)
    {
        pose.orientation = glm::normalize(glm::rotate(pose.orientation, glm::radians(-3.0f), glm::vec3(0.f, 1.f, 0.f)));
        physicsWorld->move_kinematic(physicsEntities[1].physicsBodyId, pose);
    }

    if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS) {
        if (isGrounded && currentAnim != PlayerAnimation::Jumping && currentAnim != PlayerAnimation::Landing && currentTime - lastJumpTime > .5f) {
            physicsWorld->add_impulse(physicsEntities[1].physicsBodyId, glm::vec3(0.f, 10000.f, 0.f));
            isGrounded = false;
            lastJumpTime = currentTime;
        }
        if (currentAnim != PlayerAnimation::Jumping) {
            currentAnim = PlayerAnimation::Jumping;
            animStartTime = currentTime;
        }
    }

    if (!wasGrounded && isGrounded) {
        currentAnim = PlayerAnimation::Landing;
        animStartTime = currentTime;
    } else if (!isGrounded && currentAnim != PlayerAnimation::Jumping) {
        currentAnim = PlayerAnimation::Jumping;
        animStartTime = currentTime;
    }

    bool isMoving = false;
    bool isSprinting = (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS || glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS);

    if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS)
    {   
        glm::vec3 dir = pose.orientation * glm::vec3(0.f, 0.f, 1.f);
        float speed = isSprinting ? 12.5f : 10.0f;
        
        glm::vec3 rayOrigin = pose.position + dir * 0.5f + glm::vec3(0.0f, 0.5f, 0.0f);
        glm::vec3 rayDir = glm::vec3(0.0f, -1.0f, 0.0f);
        
        float hitDist;
        glm::vec3 hitNormal;
        JPH::BodyID hitBody;
        bool groundAhead = physicsWorld->raycast(rayOrigin, rayDir, 1.5f, hitDist, hitNormal, hitBody, physicsEntities[1].physicsBodyId);

        glm::vec3 currentVel = physicsWorld->get_linear_velocity(physicsEntities[1].physicsBodyId);
        if (groundAhead && currentAnim != PlayerAnimation::Landing && currentAnim != PlayerAnimation::Jumping)
        {
            physicsWorld->set_linear_velocity(physicsEntities[1].physicsBodyId, glm::vec3(dir.x * speed, currentVel.y, dir.z * speed));
            isMoving = true;
        }
        else
        {
            physicsWorld->set_linear_velocity(physicsEntities[1].physicsBodyId, glm::vec3(currentVel.x, currentVel.y, currentVel.z));
        }
    }
    else if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS)
    {
        PhysicsPose pose = physicsWorld->get_body_pose(physicsEntities[1].physicsBodyId);
        glm::vec3 dir = pose.orientation * glm::vec3(0.f, 0.f, 1.f);
        float speed = isSprinting ? 12.5f : 10.0f;
        
        glm::vec3 currentVel = physicsWorld->get_linear_velocity(physicsEntities[1].physicsBodyId);
        physicsWorld->set_linear_velocity(physicsEntities[1].physicsBodyId, glm::vec3(-dir.x * speed, currentVel.y, -dir.z * speed));
        isMoving = true;
    }
    else if (glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS)
    {
        PhysicsPose initialPose;
        initialPose.position = glm::vec3(0.0f, 1.0f, 0.0f);
        initialPose.orientation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
        physicsWorld->move_kinematic(physicsEntities[1].physicsBodyId, initialPose);
        physicsWorld->set_linear_velocity(physicsEntities[1].physicsBodyId, glm::vec3(0.0f));

        currentAnim = PlayerAnimation::BindPose;
    }
    else {
        glm::vec3 currentVel = physicsWorld->get_linear_velocity(physicsEntities[1].physicsBodyId);
        float dampingFactor = 0.8f;
        physicsWorld->set_linear_velocity(physicsEntities[1].physicsBodyId, glm::vec3(currentVel.x * dampingFactor, currentVel.y, currentVel.z * dampingFactor));
    }

    PlayerAnimation targetMoveAnim = isSprinting ? PlayerAnimation::Sprinting : PlayerAnimation::Running;

    if (isGrounded) {
        if (currentAnim == PlayerAnimation::Landing) {
            float animDuration = models[1]->getAnimationDuration(static_cast<uint32_t>(PlayerAnimation::Landing));
            if (currentTime - animStartTime >= animDuration) {
                if (isMoving) {
                    currentAnim = targetMoveAnim;
                    animStartTime = currentTime;
                } else {
                    currentAnim = PlayerAnimation::BindPose;
                }
            }
        } else if (isMoving) {
            if (currentAnim != targetMoveAnim) {
                currentAnim = targetMoveAnim;
                animStartTime = currentTime;
            }
        } else if (currentAnim != PlayerAnimation::Landing) {
            currentAnim = PlayerAnimation::BindPose;
        }
    }

    float elapsedTime = currentTime - animStartTime;
    switch (currentAnim) {
        case PlayerAnimation::Jumping:
            models[1]->updateAnimation(static_cast<uint32_t>(PlayerAnimation::Jumping), elapsedTime, false);
            break;
        case PlayerAnimation::Landing:
            models[1]->updateAnimation(static_cast<uint32_t>(PlayerAnimation::Landing), elapsedTime, false);
            break;
        case PlayerAnimation::Running:
            models[1]->updateAnimation(static_cast<uint32_t>(PlayerAnimation::Running), elapsedTime, true);
            break;
        case PlayerAnimation::Sprinting:
            models[1]->updateAnimation(static_cast<uint32_t>(PlayerAnimation::Sprinting), elapsedTime, true);
            break;
        case PlayerAnimation::BindPose:
        default:
            models[1]->resetToBindPose();
            break;
    }

    wasGrounded = isGrounded;
}

void loadEntityFromLevel(EntityConfig conf, PhysicsWorld& world, int& playerIndex, vk::raii::Device& device, vk::raii::PhysicalDevice& physDev, vk::raii::CommandPool& comPool, vk::raii::Queue& graphQueue, TextureManager& textMgr, SkinMgr& skin) {
    std::unique_ptr<GltfModel> localModel = std::make_unique<GltfModel>(
        conf.modelPath,
        device,
        physDev,
        comPool,
        graphQueue,
        textMgr,
        skin);

    models.push_back(std::move(localModel));

    if (conf.isPlayer) {
        if (playerIndex != ~0) throw std::runtime_error("Cannot accept more than 2 players");

        playerIndex = models.size();
    }

    models.back()->setStaticTransform(glm::scale_slow(glm::mat4(1.f), glm::vec3(conf.transform.scale)) * glm::mat4_cast(conf.transform.rotation));

    std::string motionSetting = conf.physics.motionType;
    JPH::EMotionType motion = (motionSetting == "Static") ? JPH::EMotionType::Static : (motionSetting == "Kinematic" ? JPH::EMotionType::Kinematic : JPH::EMotionType::Dynamic);

    std::string layerSetting = conf.physics.layer;
    JPH::ObjectLayer layer = (layerSetting == "NON_MOVING") ? Layers::NON_MOVING : Layers::MOVING;

    std::string shapeTypeSetting = conf.physics.shapeType;

    JPH::ShapeSettings* shapeSettings = nullptr;
    if (shapeTypeSetting == "Mesh") {
        shapeSettings = models.back()->getMeshShape();
    } else if (shapeTypeSetting == "Convex" || shapeTypeSetting == "Hull" || shapeTypeSetting == "ConvexHull") {
        shapeSettings = models.back()->getConvexHull();
    } else if (shapeTypeSetting == "Box") {
        shapeSettings = models.back()->getBoxShape();
    } else {
        throw std::runtime_error("Unknown shape type: " + shapeTypeSetting);
    }

    JPH::BodyCreationSettings settings(
        shapeSettings,
        JPH::RVec3(conf.transform.position.x, conf.transform.position.y, conf.transform.position.z),
        JPH::Quat(conf.transform.rotation.x, conf.transform.rotation.y, conf.transform.rotation.z, conf.transform.rotation.w),
        motion,
        layer
    );

    settings.mFriction = conf.physics.friction;
    settings.mRestitution = conf.physics.restitution;
    settings.mLinearDamping = conf.physics.linearDamping;
    settings.mMotionQuality = (conf.physics.motionQuality == "LinearCast") ? JPH::EMotionQuality::LinearCast : JPH::EMotionQuality::Discrete;

    if (!conf.physics.allowedDOFs.empty()) {
        JPH::EAllowedDOFs dofs = JPH::EAllowedDOFs::None;
        for (const auto& dofStr : conf.physics.allowedDOFs) {
            if (dofStr == "TranslationX") dofs |= JPH::EAllowedDOFs::TranslationX;
            else if (dofStr == "TranslationY") dofs |= JPH::EAllowedDOFs::TranslationY;
            else if (dofStr == "TranslationZ") dofs |= JPH::EAllowedDOFs::TranslationZ;
            else if (dofStr == "RotationX") dofs |= JPH::EAllowedDOFs::RotationX;
            else if (dofStr == "RotationY") dofs |= JPH::EAllowedDOFs::RotationY;
            else if (dofStr == "RotationZ") dofs |= JPH::EAllowedDOFs::RotationZ;
            else if (dofStr == "All") dofs = JPH::EAllowedDOFs::All;
        }
        settings.mAllowedDOFs = dofs;
    }

    JPH::BodyID bodyId = world.create_body(settings);

    PhysicsEntity entity;
    entity.physicsBodyId = bodyId;
    entity.graphicModel = models.back().get();

    physicsEntities.push_back(entity);
}

void LogicEngine::loadModels(VulkanApp* app) {
    /*camera.Position = glm::vec3(0.f, 2.f, 0.f);
    models.push_back(std::make_unique<GltfModel>(
        "res/models/world.glb",
        *app->getDevicePtr(),
        *app->getPhysDevicePtr(),
        *app->getCommandPoolPtr(),
        *app->getGraphicsQueuePtr(),
        *app->getTextMgrPtr(),
        *app->getSkinMgrPtr()));

    models.back()->setStaticTransform(glm::scale_slow(glm::mat4(1.f), glm::vec3(10.f)));

    JPH::BodyCreationSettings floorSettings(
        models.back()->getMeshShape(), 
        JPH::RVec3(0.0f, -10.0f, 0.0f),
        JPH::Quat::sIdentity(), 
        JPH::EMotionType::Static, 
        Layers::NON_MOVING
    );
    floorSettings.mFriction = 1.0f;
    JPH::BodyID floorBodyId = app->physicsWorld->create_body(floorSettings);

    PhysicsEntity floorEntity;
    floorEntity.physicsBodyId = floorBodyId;
    floorEntity.graphicModel = models.back().get();

    physicsEntities.push_back(floorEntity);

    models.push_back(std::make_unique<GltfModel>(
        "res/models/man.glb",
        *app->getDevicePtr(),
        *app->getPhysDevicePtr(),
        *app->getCommandPoolPtr(),
        *app->getGraphicsQueuePtr(),
        *app->getTextMgrPtr(),
        *app->getSkinMgrPtr()));
    
    models.back()->setStaticTransform(glm::scale(glm::mat4(1.f), glm::vec3(.5f)));

    JPH::BodyCreationSettings sphereSettings(
        models.back()->getBoxShape(),
        JPH::RVec3(-3.f, 3.f, -3.f),
        JPH::Quat::sIdentity(),
        JPH::EMotionType::Dynamic,
        Layers::MOVING
    );
    sphereSettings.mMotionQuality = JPH::EMotionQuality::LinearCast;
    sphereSettings.mFriction = 1.0f;
    sphereSettings.mLinearDamping = 0.1f;
    sphereSettings.mAllowedDOFs = JPH::EAllowedDOFs::TranslationX | JPH::EAllowedDOFs::TranslationY | JPH::EAllowedDOFs::TranslationZ;
    JPH::BodyID sphereBodyId = app->physicsWorld->create_body(sphereSettings);

    PhysicsEntity sphereEntity;
    sphereEntity.physicsBodyId = sphereBodyId;
    sphereEntity.graphicModel = models.back().get();

    physicsEntities.push_back(sphereEntity);*/

    LevelLoader::loadFromFile("res/levels/0.json", levelData);

    // Environment
    // -----------
    // Camera
    camera.lookAt(levelData.environment.camera.lookAt);
    camera.Zoom = levelData.environment.camera.fov;
    camera.Position = levelData.environment.camera.position;

    // Sun
    hourUTC = levelData.environment.sun.hourUTC;
    year = levelData.environment.sun.year;
    month = levelData.environment.sun.month;
    day = levelData.environment.sun.day;

    latitude = levelData.environment.sun.latitude;
    longitude = levelData.environment.sun.longitude;

    autoTimeCycle = levelData.environment.sun.autoTimeCycle;
    timeCycleSpeed = levelData.environment.sun.timeCycleSpeed;

    // Graphisms
    enableRtao = levelData.environment.graphics.enableRtao;
    enableReflections = levelData.environment.graphics.enableReflections;

    // Load entities
    for (int i = 0; i < levelData.entities.size(); i++) {
        loadEntityFromLevel(
            levelData.entities[i], 
            *app->physicsWorld,
            playerIndex,
            *app->getDevicePtr(),
            *app->getPhysDevicePtr(),
            *app->getCommandPoolPtr(),
            *app->getGraphicsQueuePtr(),
            *app->getTextMgrPtr(),
            *app->getSkinMgrPtr()            
        );
    }
}

void LogicEngine::nextFrame(const VulkanApp* app) {
    // Paramètres de suivi de caméra 3ème personne
    constexpr float cameraDistance = 6.0f;       // Distance derrière le joueur
    constexpr float cameraHeight = 1.5f;         // Hauteur au-dessus du joueur
    constexpr float targetHeightOffset = 1.5f;   // Hauteur du point de regard
    constexpr float posSmoothSpeed = 10.0f;      // Vitesse d'amortissement de la position
    constexpr float targetSmoothSpeed = 15.0f;   // Vitesse d'amortissement du regard

    static glm::vec3 currentCamPos(0.0f);
    static glm::vec3 currentTargetPos(0.0f);
    static bool isFirstFrame = true;

    // Récupération de la pose et de la vitesse du joueur depuis la physique
    PhysicsPose pose = app->physicsWorld->get_body_pose(physicsEntities[1].physicsBodyId);
    glm::vec3 playerPos = pose.position;
    glm::vec3 playerForward = pose.orientation * glm::vec3(0.f, 0.f, 1.f);

    glm::vec3 vel = app->physicsWorld->get_linear_velocity(physicsEntities[1].physicsBodyId);
    float playerSpeed = glm::length(glm::vec2(vel.x, vel.z));

    // La direction de la caméra est pilotée par la souris via camera.Front
    glm::vec3 camFront = camera.Front;

    // Recentrage automatique de la caméra derrière le personnage lorsqu'il se déplace
    if (playerSpeed > 0.5f)
    {
        constexpr float autoAlignSpeed = 4.0f; // Vitesse de recentrage de la caméra derrière le joueur
        camFront = glm::normalize(glm::mix(camFront, playerForward, std::min(1.0f, autoAlignSpeed * app->deltaTime)));
        camera.lookAt(playerPos + camFront * 10.0f + glm::vec3(0.f, cameraHeight, 0.f));
    }

    // Positions idéales (sans lissage)
    glm::vec3 idealTargetPos = playerPos + glm::vec3(0.0f, targetHeightOffset, 0.0f);
    // La caméra se place derrière le point visé sur la sphère d'orbite (-camFront * distance)
    glm::vec3 idealCamPos = idealTargetPos - camFront * cameraDistance;

    // Initialisation à la première frame (évite les téléportations)
    if (isFirstFrame)
    {
        currentCamPos = idealCamPos;
        currentTargetPos = idealTargetPos;
        isFirstFrame = false;
    }
    else
    {
        // Amortissement exponentiel fluide indépendant du framerate
        float posAlpha = 1.0f - std::exp(-posSmoothSpeed * app->deltaTime);
        float targetAlpha = 1.0f - std::exp(-targetSmoothSpeed * app->deltaTime);

        currentCamPos = glm::mix(currentCamPos, idealCamPos, posAlpha);
        currentTargetPos = glm::mix(currentTargetPos, idealTargetPos, targetAlpha);
    }

    // Mise à jour de la caméra
    camera.Position = currentCamPos;
    camera.lookAt(currentTargetPos);

    // 24 h en 10 minutes (600 secondes)
    //hourUTC += (24.0f / 600.0f) * app->deltaTime;
}