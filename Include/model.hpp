#pragma once

#if defined(__INTELLISENSE__) || !defined(USE_CPP20_MODULES)
#define VULKAN_HPP_NO_STRUCT_CONSTRUCTORS
#include <vulkan/vulkan_raii.hpp>
#else
#define VULKAN_HPP_NO_STRUCT_CONSTRUCTORS
import vulkan_hpp;
#endif

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <array>
#include <vector>
#include <string>
#include <optional>

#include "vertex.hpp"

#ifndef STB_IMAGE
#include "stb_image.h"
#endif

#ifndef TINYGLTF
#include <tiny_gltf.h>
#endif

class TextureManager;

class GltfMaterial {
public:
    GltfMaterial(const tinygltf::Model& root, tinygltf::Material material, bool hasNormals, TextureManager& textureManager);
    GltfMaterial() = default;
    ~GltfMaterial() = default;

    void bind(vk::raii::CommandBuffer& commandBuffer, vk::raii::PipelineLayout& pipelineLayout, glm::mat4 modelMatrix) const;

private:
    uint8_t features; // Bitfield for features for this material

    double basecolor[3];
    double metallic_factor;
    double roughness_factor;

    std::optional<uint32_t> baseColorTextureIndex;
    std::optional<uint32_t> metallicRoughnessTextureIndex;
};

class GltfPrimitive {
public:
    GltfPrimitive(const tinygltf::Model& root, uint32_t primfirstIndex, uint32_t primIndexCount, vk::DeviceSize primByteOffset, tinygltf::Material material, bool hasNormals, vk::VertexInputBindingDescription2EXT binding, std::vector<vk::VertexInputAttributeDescription2EXT> attributes, TextureManager& textureManager);

    void draw(vk::raii::CommandBuffer& commandBuffer, vk::raii::PipelineLayout& pipelineLayout, vk::raii::Buffer& globalVertexBuffer, glm::mat4 modelMatrix) const;

private:
    GltfMaterial material;

    uint32_t firstIndex;   // Index de départ dans le globalIndexBuffer
    uint32_t indexCount;   // Nombre d'indices à dessiner
    vk::DeviceSize byteOffset;
    vk::VertexInputBindingDescription2EXT vertexBindingDescription;
    std::vector<vk::VertexInputAttributeDescription2EXT> vertexAttributeDescriptions;
};

class GltfMesh {
public:
    // Reçoit simplement les primitives déjà construites
    GltfMesh() = default;
    void addPrimitive(GltfPrimitive&& primitive) {
        primitives.push_back(std::move(primitive));
    }
    
    void draw(vk::raii::CommandBuffer& commandBuffer, vk::raii::PipelineLayout& pipelineLayout, vk::raii::Buffer& globalVertexBuffer, glm::mat4 modelMatrix) const {
        for (auto& primitive : primitives) {
            primitive.draw(commandBuffer, pipelineLayout, globalVertexBuffer, modelMatrix);
        }
    }
private:
    std::vector<GltfPrimitive> primitives;
};

class GltfModel;

class GltfNode {
public:
    // Constructeur simple sans Vulkan
    GltfNode(GltfModel& model, tinygltf::Model& root, tinygltf::Node node, glm::mat4 parent_node_transform);

    void draw(vk::raii::CommandBuffer& commandBuffer, vk::raii::PipelineLayout& pipelineLayout, glm::mat4 parentMatrix, vk::raii::Buffer& globalVertexBuffer) const {
        glm::mat4 globalTransform = parentMatrix * node_transform;

        if (mesh) {
            // Exemple : Envoi de la matrice finale via Push Constants avant de draw le mesh
            mesh->draw(commandBuffer, pipelineLayout, globalVertexBuffer, globalTransform);
        }

        for (auto& child : children) {
            child.draw(commandBuffer, pipelineLayout, globalTransform, globalVertexBuffer);
        }
    }

private:
    const GltfMesh* mesh;
    std::vector<GltfNode> children;
    glm::mat4 node_transform;
};

class GltfModel {
public:
    GltfModel(const std::string& path, vk::raii::Device& device, vk::raii::PhysicalDevice& physicalDevice, vk::raii::CommandPool& commandPool, vk::raii::Queue& graphicsQueue, TextureManager& textureManager);
    ~GltfModel() = default;

    // Ajout du pipelineLayout pour mettre à jour les transformations des nodes lors du dessin
    void draw(vk::raii::CommandBuffer& commandBuffer, vk::raii::PipelineLayout& pipelineLayout) {
        // 1. Bind des buffers globaux UNE SEULE FOIS pour tout le modèle
        // commandBuffer.bindVertexBuffers(0, {*globalVertexBuffer}, {0}); // Supprimé, géré par primitive
        commandBuffer.bindIndexBuffer(*globalIndexBuffer, 0, vk::IndexType::eUint32);

        // 3. On lance le dessin récursif sur les nœuds racines (root nodes)
        for (auto& node : rootNodes) {
            node.draw(commandBuffer, pipelineLayout, modelTransform, globalVertexBuffer);
        }
    }

    glm::mat4 modelTransform = glm::mat4(1.f);

    // Liste des meshes
    std::vector<GltfMesh> meshes;

private:
    std::vector<unsigned char> globalVertexData;
    std::vector<uint32_t> indices;

    vk::raii::Buffer globalVertexBuffer = nullptr;
    vk::raii::DeviceMemory globalVertexMemory = nullptr;
    
    vk::raii::Buffer globalIndexBuffer = nullptr;
    vk::raii::DeviceMemory globalIndexMemory = nullptr;

    // Les nœuds racines du glTF
    std::vector<GltfNode> rootNodes;

    vk::raii::Device* device = nullptr;
    vk::raii::PhysicalDevice *physicalDevice = nullptr;
    vk::raii::CommandPool *commandPool = nullptr;
    vk::raii::Queue *graphicsQueue = nullptr;

    // Méthodes internes de génération
    void createVertexBuffer();
    void createIndexBuffer();
};