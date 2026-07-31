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

#include <vector>
#include <string>
#include <optional>
#include <limits>

#include "vertex.hpp"

#ifndef STB_IMAGE
#include "stb_image.h"
#endif

#ifndef TINYGLTF
#include <tiny_gltf.h>
#endif

#include <Jolt/Jolt.h>
#include <Jolt/Physics/Collision/Shape/ConvexHullShape.h>
#include <Jolt/Physics/Collision/Shape/MeshShape.h>
#include <Jolt/Physics/Collision/Shape/BoxShape.h>
#include <Jolt/Physics/Collision/Shape/RotatedTranslatedShape.h>

#include "skin.hh"

class TextureManager;

struct ModelPushConstants {
    glm::mat4 model;
    glm::mat4 prevModel;
    uint32_t albedoTextureIndex;
    uint32_t rmTextureIndex;
    uint32_t padding[2]; // Align baseColor to 16 bytes
    glm::vec4 baseColor;
    glm::vec4 emissiveColor;
    float roughness;
    float metallic;
    uint32_t activeAttributes;
    uint32_t padding2; // Pad to multiple of 16 bytes for safety
};

class GltfMaterial {
public:
    GltfMaterial(const tinygltf::Model& root, tinygltf::Material material, bool hasNormals, TextureManager& textureManager, const std::string& modelPath);
    GltfMaterial() = default;
    ~GltfMaterial() = default;

    void bind(vk::raii::CommandBuffer& commandBuffer, vk::raii::PipelineLayout& pipelineLayout, glm::mat4 modelMatrix, glm::mat4 prevModelMatrix) const;

private:
    uint8_t features; // Bitfield for features for this material

    double basecolor[3];
    double metallic_factor;
    double roughness_factor;
    double transmission_factor;
    double emissive_factor[3];

    std::optional<uint32_t> baseColorTextureIndex;
    std::optional<uint32_t> metallicRoughnessTextureIndex;

public:
    uint32_t getMaterialIndex() const { return baseColorTextureIndex.value_or(0); }
    glm::vec4 getBaseColor() const { return glm::vec4(basecolor[0], basecolor[1], basecolor[2], 1.0f); }
    float getMetallic() const { return metallic_factor; }
    float getRoughness() const { return roughness_factor; }
    float getTransmission() const { return transmission_factor; }
    glm::vec3 getEmissive() const { return glm::vec3(emissive_factor[0], emissive_factor[1], emissive_factor[2]); }
    uint32_t getActiveAttributes() const { return features; }
};

class GltfPrimitive {
public:
    GltfPrimitive(const tinygltf::Model& root, uint32_t primfirstIndex, uint32_t primIndexCount, uint32_t primVertexCount, vk::DeviceSize primByteOffset, tinygltf::Material material, bool hasNormals, vk::VertexInputBindingDescription2EXT binding, std::vector<vk::VertexInputAttributeDescription2EXT> attributes, TextureManager& textureManager, const std::string& modelPath);

    void draw(vk::raii::CommandBuffer& commandBuffer, vk::raii::PipelineLayout& pipelineLayout, vk::raii::Buffer& globalVertexBuffer, glm::mat4 modelMatrix, glm::mat4 prevModelMatrix) const;

    uint32_t getFirstIndex() const { return firstIndex; }
    uint32_t getIndexCount() const { return indexCount; }
    vk::DeviceSize getByteOffset() const { return byteOffset; }
    uint32_t getVertexCount() const { return vertexCount; }
    uint32_t getStride() const { return vertexBindingDescription.stride; }

    uint32_t getUvOffset() const {
        for (const auto& attr : vertexAttributeDescriptions) {
            if (attr.location == 2) {
                return attr.offset;
            }
        }
        return 0;
    }
    uint32_t getNormalOffset() const {
        for (const auto& attr : vertexAttributeDescriptions) {
            if (attr.location == 1) {
                return attr.offset;
            }
        }
        return 0;
    }
    const GltfMaterial& getMaterial() const { return material; }

private:
    GltfMaterial material;

    uint32_t firstIndex;   // Index de départ dans le globalIndexBuffer
    uint32_t indexCount;   // Nombre d'indices à dessiner
    vk::DeviceSize byteOffset;
    uint32_t vertexCount;
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
    
    void draw(vk::raii::CommandBuffer& commandBuffer, vk::raii::PipelineLayout& pipelineLayout, vk::raii::Buffer& globalVertexBuffer, glm::mat4 modelMatrix, glm::mat4 prevModelMatrix) const {
        for (auto& primitive : primitives) {
            primitive.draw(commandBuffer, pipelineLayout, globalVertexBuffer, modelMatrix, prevModelMatrix);
        }
    }

     void buildBlas(vk::raii::Device& device, 
                   vk::raii::PhysicalDevice& physicalDevice, 
                   vk::raii::CommandPool& commandPool, 
                   vk::raii::Queue& graphicsQueue,
                   vk::DeviceAddress vertexBufferAddress,
                   vk::DeviceAddress indexBufferAddress);

    void updateBlas(vk::raii::CommandBuffer& commandBuffer, 
                    vk::DeviceAddress vertexBufferAddress, 
                    vk::DeviceAddress indexBufferAddress);

    vk::DeviceAddress getBlasAddress(vk::raii::Device& device) const {
        vk::AccelerationStructureDeviceAddressInfoKHR addressInfo{ .accelerationStructure = *blasHandle };
        return device.getAccelerationStructureAddressKHR(addressInfo);
    }

    const std::vector<GltfPrimitive>& getPrimitives() const { return primitives; }
    uint32_t getPrimitiveInstanceCount() const { return primitives.size(); }
private:
    std::vector<GltfPrimitive> primitives;

    vk::raii::Buffer blasBuffer = nullptr;
    vk::raii::DeviceMemory blasBufferMemory = nullptr;
    vk::raii::AccelerationStructureKHR blasHandle = nullptr;

    vk::raii::Buffer updateScratchBuffer = nullptr;
    vk::raii::DeviceMemory updateScratchMemory = nullptr;
    vk::DeviceAddress updateScratchAddress = 0;
};

class GltfModel;

class GltfNode {
public:
    // Constructeur simple sans Vulkan
    GltfNode(GltfModel& model, tinygltf::Model& root, int nodeIndex, glm::mat4 parent_node_transform);

    void draw(vk::raii::CommandBuffer& commandBuffer, vk::raii::PipelineLayout& pipelineLayout, glm::mat4 parentMatrix, glm::mat4 prevParentMatrix, vk::raii::Buffer& globalVertexBuffer, bool isSkinned, glm::mat4 rootTransform, glm::mat4 prevRootTransform) const {
        glm::mat4 prevGlobalTransform = prevParentMatrix * node_transform;
        glm::mat4 globalTransform = parentMatrix * node_transform;

        if (mesh) {
            glm::mat4 renderTransform = isSkinned ? rootTransform : globalTransform;
            glm::mat4 prevRenderTransform = isSkinned ? prevRootTransform : prevGlobalTransform;
            mesh->draw(commandBuffer, pipelineLayout, globalVertexBuffer, renderTransform, prevRenderTransform);
        }

        for (auto& child : children) {
            child.draw(commandBuffer, pipelineLayout, globalTransform, prevGlobalTransform, globalVertexBuffer, isSkinned, rootTransform, prevRootTransform);
        }
    }

    void populateTlasInstances(std::vector<vk::AccelerationStructureInstanceKHR>& instances, 
                            std::vector<InstanceData>& instanceData,
                            vk::raii::Device& device, 
                            glm::mat4 parentMatrix,
                            uint32_t& customIndexOffset,
                            vk::DeviceAddress vAddr,
                            vk::DeviceAddress iAddr,
                            bool isSkinned,
                            glm::mat4 rootTransform) const ;

    uint32_t getMeshInstanceCount() const {
        uint32_t count = mesh ? 1 : 0;
        for (auto& child : children) {
            count += child.getMeshInstanceCount();
        }
        return count;
    }

    uint32_t getPrimitiveInstanceCount() const {
        uint32_t count = mesh ? mesh->getPrimitiveInstanceCount() : 0;
        for (auto& child : children) {
            count += child.getPrimitiveInstanceCount();
        }
        return count;
    }

    void collectPhysicsVertices(const std::vector<unsigned char>& globalVertexData, JPH::Array<JPH::Vec3>& outPositions, glm::mat4 parentMatrix) const;
    void collectPhysicsMeshData(const std::vector<unsigned char>& globalVertexData, const std::vector<uint32_t>& globalIndices, JPH::VertexList& outVertices, JPH::IndexedTriangleList& outTriangles, glm::mat4 parentMatrix) const;

    std::vector<GltfNode>& getChildren() {
        return children;
    };
    
    const std::vector<GltfNode>& getChildren() const {
        return children;
    };

private:
    const GltfMesh* mesh;
    std::vector<GltfNode> children;
    glm::mat4 node_transform;

public:
    int gltfNodeIndex;
    glm::vec3 translation{0.0f};
    glm::quat rotation{1.0f, 0.0f, 0.0f, 0.0f};
    glm::vec3 scale{1.0f};
    glm::mat4 local_matrix{1.0f};
    bool has_matrix = false;

    glm::vec3 initialTranslation{0.0f};
    glm::quat initialRotation{1.0f, 0.0f, 0.0f, 0.0f};
    glm::vec3 initialScale{1.0f};
    glm::mat4 initialMatrix{1.0f};

    void updateNodeTransform() {
        if (has_matrix) {
            node_transform = local_matrix;
        } else {
            glm::mat4 t = glm::translate(glm::mat4(1.0f), translation);
            glm::mat4 r = glm::mat4_cast(rotation);
            glm::mat4 s = glm::scale(glm::mat4(1.0f), scale);
            node_transform = t * r * s;
        }
        for (auto& child : children) {
            child.updateNodeTransform();
        }
    }

    glm::mat4 global_transform{1.0f};

    void updateGlobalTransform(const glm::mat4& parent_global) {
        global_transform = parent_global * node_transform;
        for (auto& child : children) {
            child.updateGlobalTransform(global_transform);
        }
    }

    void resetToInitial() {
        translation = initialTranslation;
        rotation = initialRotation;
        scale = initialScale;
        local_matrix = initialMatrix;
        updateNodeTransform();
        for (auto& child : children) {
            child.resetToInitial();
        }
    }
};

struct Skeleton {
    std::vector<GltfNode*> joints;
    std::vector<glm::mat4> inverseBindMatrices;
};

enum class AnimationPathType {
    TRANSLATION,
    ROTATION,
    SCALE
};

struct AnimationChannel {
    AnimationPathType path;
    GltfNode* node;
    uint32_t samplerIndex;
};

struct AnimationSampler {
    std::vector<float> inputs;
    std::vector<glm::vec4> outputs;
    std::string interpolation;
};

struct Animation {
    std::string name;
    std::vector<AnimationSampler> samplers;
    std::vector<AnimationChannel> channels;
    float start = std::numeric_limits<float>::max();
    float end = std::numeric_limits<float>::min();
};

namespace JPH {
    class ShapeSettings;
}

class GltfModel {
public:
    GltfModel(const std::string& path, vk::raii::Device& device, vk::raii::PhysicalDevice& physicalDevice, vk::raii::CommandPool& commandPool, vk::raii::Queue& graphicsQueue, TextureManager& textureManager, SkinMgr& skinMgr);
    ~GltfModel() = default;

    JPH::ShapeSettings* getConvexHull() const;
    JPH::ShapeSettings* getMeshShape() const;
    JPH::ShapeSettings* getBoxShape() const;

    void draw(vk::raii::CommandBuffer& commandBuffer, vk::raii::PipelineLayout& pipelineLayout) {
        commandBuffer.bindIndexBuffer(*globalIndexBuffer, 0, vk::IndexType::eUint32);

        bool isSkinned = !skeletons.empty();
        vk::raii::Buffer& vertexBuffer = isSkinned ? outputVertexBuffer : inputVertexBuffer;
        glm::mat4 rootMatrix = modelTransform * staticTransform;
        glm::mat4 prevRootMatrix = previousModelTransform * staticTransform;

        for (auto& node : rootNodes) {
            node.draw(commandBuffer, pipelineLayout, rootMatrix, prevRootMatrix, vertexBuffer, isSkinned, rootMatrix, prevRootMatrix);
        }
    }

    void populateTlasInstances(std::vector<vk::AccelerationStructureInstanceKHR>& instances, 
                            std::vector<InstanceData>& instanceData,
                            vk::raii::Device& device,
                            uint32_t& customIndexOffset) const 
    {
        bool isSkinned = !skeletons.empty();
        const vk::raii::Buffer& vertexBuffer = isSkinned ? outputVertexBuffer : inputVertexBuffer;

        vk::DeviceAddress vAddr = 0;
        if (*vertexBuffer) {
            vk::BufferDeviceAddressInfo vInfo{.buffer = *vertexBuffer};
            vAddr = device.getBufferAddress(vInfo);
        }
        
        vk::DeviceAddress iAddr = 0;
        if (*globalIndexBuffer) {
            vk::BufferDeviceAddressInfo iInfo{.buffer = *globalIndexBuffer};
            iAddr = device.getBufferAddress(iInfo);
        }

        glm::mat4 rootMatrix = modelTransform * staticTransform;

        for (auto& node : rootNodes) {
            node.populateTlasInstances(instances, instanceData, device, rootMatrix, customIndexOffset, vAddr, iAddr, isSkinned, rootMatrix);
        }
    }

    uint32_t getMeshInstanceCount() const {
        uint32_t count = 0;
        for (auto& node : rootNodes) {
            count += node.getMeshInstanceCount();
        }
        return count;
    }

    uint32_t getPrimitiveInstanceCount() const {
        uint32_t count = 0;
        for (auto& node : rootNodes) {
            count += node.getPrimitiveInstanceCount();
        }
        return count;
    }

    void setModelTransform(glm::mat4 newModelTransform) {
        previousModelTransform = modelTransform;
        modelTransform = newModelTransform;
    }

    void setStaticTransform(glm::mat4 transform) {
        if (staticTransform != glm::mat4(1.f)) throw std::runtime_error("Cannot modify static transform twice");
        staticTransform = transform; 
    }

    void updateAnimation(uint32_t index, float time, bool loop = true);
    float getAnimationDuration(uint32_t index) const;

    void resetToBindPose();

    void updateBlas(vk::raii::CommandBuffer& commandBuffer, vk::raii::Device& device);

    // Liste des meshes
    std::vector<GltfMesh> meshes;

    std::vector<Skeleton> skeletons;
    std::vector<Animation> animations;

    vk::raii::DescriptorPool computeDescriptorPool = nullptr;
    SkinComputeResources skin;

    void populateLinearNodes(GltfNode& node) {
        if (node.gltfNodeIndex >= 0 && node.gltfNodeIndex < linearNodes.size()) {
            linearNodes[node.gltfNodeIndex] = &node;
        }

        for (auto& child : node.getChildren()) {
            populateLinearNodes(child);
        }
    }

private:
    glm::mat4 staticTransform = glm::mat4(1.f);

    std::vector<unsigned char> globalVertexData;
    std::vector<uint32_t> indices;

    vk::raii::Buffer inputVertexBuffer = nullptr;
    vk::raii::DeviceMemory inputVertexBufferMemory = nullptr;

    vk::raii::Buffer outputVertexBuffer = nullptr;
    vk::raii::DeviceMemory outputVertexBufferMemory = nullptr;

    vk::raii::Buffer joinMatrixBuffer = nullptr;
    vk::raii::DeviceMemory joinMatrixBufferMemory = nullptr;
    
    vk::raii::Buffer globalIndexBuffer = nullptr;
    vk::raii::DeviceMemory globalIndexMemory = nullptr;

    // Les nœuds racines et globaux du glTF
    std::vector<GltfNode> rootNodes;
    std::vector<GltfNode*> linearNodes;

    vk::raii::Device* device = nullptr;
    vk::raii::PhysicalDevice *physicalDevice = nullptr;
    vk::raii::CommandPool *commandPool = nullptr;
    vk::raii::Queue *graphicsQueue = nullptr;

    glm::mat4 modelTransform = glm::mat4(1.f);
    glm::mat4 previousModelTransform = glm::mat4(1.f);

    // Méthodes internes de génération
    void createVertexBuffer();
    void createIndexBuffer();
    void createJoinMatrixBuffer();

    void createComputeResources(SkinMgr& skinMgr);
};