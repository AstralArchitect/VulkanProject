#pragma once

#include <string>
#include <vector>
#include <fstream>
#include <iostream>

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

struct SunConfig {
    int year = 2026;
    int month = 9;
    int day = 23;
    double hourUTC = 8.0;
    double latitude = 48.8566;
    double longitude = 2.3522;
    bool autoTimeCycle = false;
    float timeCycleSpeed = 1.0f;
};

struct CameraConfig {
    glm::vec3 position{0.0f, 2.0f, 6.0f};
    glm::vec3 lookAt{0.0f, 1.0f, 0.0f};
    float fov = 45.0f;
};

struct GraphicsConfig {
    bool enableReflections = true;
    bool enableRtao = true;
    int ffxMaxSamples = 16;
};

struct EnvironmentConfig {
    SunConfig sun;
    CameraConfig camera;
    GraphicsConfig graphics;
};

struct EntityConfig {
    int id;
    std::string name;
    std::string modelPath;
    bool isPlayer = false;

    struct TransformConfig {
        glm::vec3 position{0.0f};
        glm::quat rotation{1.0f, 0.0f, 0.0f, 0.0f};
        glm::vec3 scale{1.0f};
    } transform;

    struct PhysicsConfig {
        bool enabled = true;
        std::string motionType = "Static";
        std::string shapeType = "Mesh";
        std::string layer = "NON_MOVING";
        float friction = 1.0f;
        float restitution = 0.0f;
        float linearDamping = 0.1f;
        std::string motionQuality = "Discrete";
        std::vector<std::string> allowedDOFs;
    } physics;
};

struct LevelData {
    std::string name;
    std::string version;
    EnvironmentConfig environment;
    std::vector<EntityConfig> entities;
};

namespace LevelLoader {
    static glm::vec3 parseVec3(const json& j, const glm::vec3& defaultVal = glm::vec3(0.0f)) {
        if (j.is_array() && j.size() >= 3) {
            return glm::vec3(j[0].get<float>(), j[1].get<float>(), j[2].get<float>());
        }
        return defaultVal;
    }

    static glm::quat parseQuat(const json& j, const glm::quat& defaultVal = glm::quat(1.0f, 0.0f, 0.0f, 0.0f)) {
        if (j.is_array() && j.size() >= 4) {
            // Expected format in JSON: [x, y, z, w]
            return glm::quat(j[3].get<float>(), j[0].get<float>(), j[1].get<float>(), j[2].get<float>());
        }
        return defaultVal;
    }

    static bool loadFromFile(const std::string& filepath, LevelData& outLevelData) {
        std::ifstream file(filepath);
        if (!file.is_open()) {
            std::cerr << "[LevelLoader] Error: Unable to open level file: " << filepath << std::endl;
            return false;
        }

        try {
            json j;
            file >> j;

            outLevelData.name = j.value("name", "Untitled Level");
            outLevelData.version = j.value("version", "1.0");

            // Environment configuration
            if (j.contains("environment")) {
                const auto& env = j["environment"];

                // Sun / Time parameters
                if (env.contains("sun")) {
                    const auto& sun = env["sun"];
                    outLevelData.environment.sun.year = sun.value("year", 2026);
                    outLevelData.environment.sun.month = sun.value("month", 9);
                    outLevelData.environment.sun.day = sun.value("day", 23);
                    outLevelData.environment.sun.hourUTC = sun.value("hourUTC", 8.0);
                    outLevelData.environment.sun.latitude = sun.value("latitude", 48.8566);
                    outLevelData.environment.sun.longitude = sun.value("longitude", 2.3522);
                    outLevelData.environment.sun.autoTimeCycle = sun.value("autoTimeCycle", false);
                    outLevelData.environment.sun.timeCycleSpeed = sun.value("timeCycleSpeed", 1.0f);
                }

                // Camera parameters
                if (env.contains("camera")) {
                    const auto& cam = env["camera"];
                    outLevelData.environment.camera.position = parseVec3(cam["position"], glm::vec3(0.0f, 2.0f, 6.0f));
                    outLevelData.environment.camera.lookAt = parseVec3(cam["lookAt"], glm::vec3(0.0f, 1.0f, 0.0f));
                    outLevelData.environment.camera.fov = cam.value("fov", 45.0f);
                }

                // Graphics parameters
                if (env.contains("graphics")) {
                    const auto& gfx = env["graphics"];
                    outLevelData.environment.graphics.enableReflections = gfx.value("enableReflections", true);
                    outLevelData.environment.graphics.enableRtao = gfx.value("enableRtao", true);
                    outLevelData.environment.graphics.ffxMaxSamples = gfx.value("ffxMaxSamples", 16);
                }
            }

            // Entities configuration
            if (j.contains("entities") && j["entities"].is_array()) {
                outLevelData.entities.clear();
                for (const auto& item : j["entities"]) {
                    EntityConfig entity;
                    entity.id = item.value("id", 0);
                    entity.name = item.value("name", "Entity");
                    entity.modelPath = item.value("modelPath", "");
                    entity.isPlayer = item.value("isPlayer", false);

                    // Transform
                    if (item.contains("transform")) {
                        const auto& t = item["transform"];
                        entity.transform.position = parseVec3(t["position"]);
                        entity.transform.rotation = parseQuat(t["rotation"]);
                        entity.transform.scale = parseVec3(t["scale"], glm::vec3(1.0f));
                    }

                    // Physics
                    if (item.contains("physics")) {
                        const auto& p = item["physics"];
                        entity.physics.enabled = p.value("enabled", true);
                        entity.physics.motionType = p.value("motionType", "Static");
                        entity.physics.shapeType = p.value("shapeType", "Mesh");
                        entity.physics.layer = p.value("layer", "NON_MOVING");
                        entity.physics.friction = p.value("friction", 1.0f);
                        entity.physics.restitution = p.value("restitution", 0.0f);
                        entity.physics.linearDamping = p.value("linearDamping", 0.1f);
                        entity.physics.motionQuality = p.value("motionQuality", "Discrete");

                        if (p.contains("allowedDOFs") && p["allowedDOFs"].is_array()) {
                            for (const auto& dof : p["allowedDOFs"]) {
                                entity.physics.allowedDOFs.push_back(dof.get<std::string>());
                            }
                        }
                    }

                    outLevelData.entities.push_back(entity);
                }
            }

            std::cout << "[LevelLoader] Level loaded successfully: " << outLevelData.name
                    << " (" << outLevelData.entities.size() << " entities)" << std::endl;
            return true;
        }
        catch (const std::exception& e) {
            std::cerr << "[LevelLoader] Exception parsing level JSON: " << e.what() << std::endl;
            return false;
        }
    }

};
