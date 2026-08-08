# 🗺️ Feuille de Route : Implémentation du Level Loader JSON

Ce document sert de guide étape par étape pour implémenter le chargement dynamique de niveaux depuis le fichier [`res/levels/0.json`](file:///c:/Users/matth/Documents/Code/VulkanProject/res/levels/0.json).

---

## 📌 Objectifs
1. Charger la configuration de l'environnement (Soleil, Caméra, Graphismes) depuis le JSON.
2. Instancier dynamiquement la liste des modèles 3D (`.glb`).
3. Configurer automatiquement les corps physiques Jolt (Statique/Dynamique, Forme Mesh/Box, Friction, Damping).
4. Découpler le chargement "hardcodé" dans `LogicEngine::loadModels()`.

---

## 🛠️ Étape 1 : Ajouter la bibliothèque JSON (`nlohmann_json`)

Dans votre fichier [`meson.build`](file:///c:/Users/matth/Documents/Code/VulkanProject/meson.build) :

1. Déclarer la dépendance `nlohmann_json` :
```meson
nlohmann_json_dep = dependency('nlohmann_json')
```
2. Ajouter `nlohmann_json_dep` dans la liste des dépendances de l'exécutable `main` :
```meson
executable(
    'main',
    sources: sources,
    include_directories: include_dir,
    dependencies: [vulkan_dep, glfw_dep, glm_dep, tinygltf_dep, jolt_dep, imgui_dep, nlohmann_json_dep],
)
```

*(Note : Si vous utilisez déjà `nlohmann/json.hpp` via une autre inclusion ou en header-only, vous pouvez directement inclure `<nlohmann/json.hpp>`).*

---

## 📂 Étape 2 : Créer la classe `LevelLoader`

Créer un header [`Include/level_loader.hh`](file:///c:/Users/matth/Documents/Code/VulkanProject/Include/level_loader.hh) et sa source [`src/level_loader.cc`](file:///c:/Users/matth/Documents/Code/VulkanProject/src/level_loader.cc).

### Structure suggérée dans `level_loader.hh` :

```cpp
#pragma once

#include <string>
#include <vector>
#include <glm/glm.hpp>
#include <nlohmann/json.hpp>

class VulkanApp;
class LogicEngine;

struct EntityConfig {
    std::string id;
    std::string name;
    std::string modelPath;
    bool isPlayer = false;
    
    struct Transform {
        glm::vec3 position{0.0f};
        glm::vec3 rotationEuler{0.0f}; // ou glm::quat
        glm::vec3 scale{1.0f};
    } transform;

    struct PhysicsConfig {
        bool enabled = true;
        std::string motionType; // "Static" ou "Dynamic"
        std::string shapeType;  // "Mesh" ou "Box"
        std::string layer;      // "NON_MOVING" ou "MOVING"
        float friction = 1.0f;
        float linearDamping = 0.1f;
        float restitution = 0.0f;
        std::string motionQuality; // "Discrete" ou "LinearCast"
    } physics;
};

struct LevelData {
    std::string name;
    std::string version;

    // Soleil / Temps
    int year = 2026;
    int month = 9;
    int day = 23;
    double hourUTC = 8.0;
    double latitude = 48.8566;
    double longitude = 2.3522;
    bool autoTimeCycle = false;
    float timeCycleSpeed = 1.0f;

    // Caméra
    glm::vec3 cameraPosition{0.0f, 2.0f, 6.0f};
    glm::vec3 cameraLookAt{0.0f, 1.0f, 0.0f};
    float cameraFov = 45.0f;

    // Entités
    std::vector<EntityConfig> entities;
};

class LevelLoader {
public:
    static bool loadFromFile(const std::string& filepath, LevelData& outLevelData);
};
```

---

## ⚙️ Étape 3 : Parser les données JSON dans `src/level_loader.cc`

Utiliser des fonctions utilitaires simples pour convertir les tableaux JSON en `glm::vec3` :

```cpp
#include "level_loader.hh"
#include <fstream>
#include <iostream>

using json = nlohmann::json;

static glm::vec3 parseVec3(const json& j, const glm::vec3& defaultVal = glm::vec3(0.0f)) {
    if (j.is_array() && j.size() >= 3) {
        return glm::vec3(j[0].get<float>(), j[1].get<float>(), j[2].get<float>());
    }
    return defaultVal;
}

bool LevelLoader::loadFromFile(const std::string& filepath, LevelData& outLevelData) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "[LevelLoader] Impossible d'ouvrir le fichier : " << filepath << std::endl;
        return false;
    }

    json j;
    file >> j;

    outLevelData.name = j.value("name", "Untitled Level");
    outLevelData.version = j.value("version", "1.0");

    // 1. Environnement & Soleil
    if (j.contains("environment")) {
        auto& env = j["environment"];
        if (env.contains("sun")) {
            auto& sun = env["sun"];
            outLevelData.year = sun.value("year", 2026);
            outLevelData.month = sun.value("month", 9);
            outLevelData.day = sun.value("day", 23);
            outLevelData.hourUTC = sun.value("hourUTC", 8.0);
            outLevelData.latitude = sun.value("latitude", 48.8566);
            outLevelData.longitude = sun.value("longitude", 2.3522);
            outLevelData.autoTimeCycle = sun.value("autoTimeCycle", false);
            outLevelData.timeCycleSpeed = sun.value("timeCycleSpeed", 1.0f);
        }

        if (env.contains("camera")) {
            auto& cam = env["camera"];
            outLevelData.cameraPosition = parseVec3(cam["position"], glm::vec3(0.0f, 2.0f, 6.0f));
            outLevelData.cameraLookAt = parseVec3(cam["lookAt"], glm::vec3(0.0f, 1.0f, 0.0f));
            outLevelData.cameraFov = cam.value("fov", 45.0f);
        }
    }

    // 2. Entités
    if (j.contains("entities") && j["entities"].is_array()) {
        for (const auto& item : j["entities"]) {
            EntityConfig entity;
            entity.id = item.value("id", "");
            entity.name = item.value("name", "Entity");
            entity.modelPath = item.value("modelPath", "");
            entity.isPlayer = item.value("isPlayer", false);

            if (item.contains("transform")) {
                auto& t = item["transform"];
                entity.transform.position = parseVec3(t["position"]);
                entity.transform.scale = parseVec3(t["scale"], glm::vec3(1.0f));
            }

            if (item.contains("physics")) {
                auto& p = item["physics"];
                entity.physics.enabled = p.value("enabled", true);
                entity.physics.motionType = p.value("motionType", "Static");
                entity.physics.shapeType = p.value("shapeType", "Mesh");
                entity.physics.layer = p.value("layer", "NON_MOVING");
                entity.physics.friction = p.value("friction", 1.0f);
                entity.physics.linearDamping = p.value("linearDamping", 0.1f);
                entity.physics.motionQuality = p.value("motionQuality", "Discrete");
            }

            outLevelData.entities.push_back(entity);
        }
    }

    std::cout << "[LevelLoader] Niveau charge avec succes : " << outLevelData.name 
              << " (" << outLevelData.entities.size() << " entites)" << std::endl;
    return true;
}
```

---

## 🔄 Étape 4 : Refactoriser `LogicEngine::loadModels()`

Dans [`src/logic_engine.cc`](file:///c:/Users/matth/Documents/Code/VulkanProject/src/logic_engine.cc) :

1. Appeler `LevelLoader::loadFromFile("res/levels/0.json", levelData);`
2. Affecter les paramètres du Soleil & de la Caméra.
3. Boucler sur `levelData.entities` :
   - Charger le modèle `GltfModel` avec sa matrice de transform initiale (`entity.transform`).
   - Construire la forme physique Jolt dynamique (`models.back()->getMeshShape()` ou `models.back()->getBoxShape()`).
   - Créer le body Jolt (`app->physicsWorld->create_body(...)`).
   - Enregistrer dans `physicsEntities`.

---

## 🚀 Étape 5 : Mise à jour de `meson.build`

Ajouter `cpp_dir + 'level_loader.cc'` dans la liste `cpp_sources` du [`meson.build`](file:///c:/Users/matth/Documents/Code/VulkanProject/meson.build).
