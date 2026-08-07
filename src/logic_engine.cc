#include "logic_engine.hh"

#include "jolt_physics.hpp"
#include <Jolt/Physics/Body/BodyCreationSettings.h>
#include <Jolt/Physics/Collision/Shape/BoxShape.h>

LogicEngine::LogicEngine() {
    camera = Camera(
        glm::vec3(2.0f, 2.0f, 6.0f), // Position
        glm::vec3(0.0f, 1.0f, 0.0f)  // World Up
    );

    camera.lookAt(glm::vec3(0.f, 1.f, 0.f));
}

void LogicEngine::movment(GameMovment movement, float deltaTime) {

}

void LogicEngine::loadModels(VulkanApp* app) {
    camera.Position = glm::vec3(0.f, 2.f, 0.f);
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

    physicsEntities.push_back(sphereEntity);
}

void LogicEngine::nextFrame(const VulkanApp* app) {
    // Paramètres de suivi de caméra 3ème personne
    constexpr float cameraDistance = 4.0f;       // Distance derrière le joueur
    constexpr float cameraHeight = 1.5f;         // Hauteur au-dessus du joueur
    constexpr float targetHeightOffset = 1.2f;   // Hauteur du point de regard (tête/buste)
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
        camera.lookAt(playerPos + camFront * 10.0f);
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