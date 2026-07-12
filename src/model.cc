#include <iostream>
#include <optional>

#include "model.hpp"

#include "vulkanUtils.hpp"

void GltfPrimitive::draw(vk::raii::CommandBuffer& commandBuffer, vk::raii::Buffer& globalVertexBuffer) const {
    
    // 1. On donne le format dynamique au pipeline (l'étape 4 de mon précédent message)
    commandBuffer.setVertexInputEXT(vertexBindingDescription, vertexAttributeDescriptions);

    // 2. On lie le Mega-Buffer MAIS en disant à Vulkan de commencer à lire exactement au byteOffset de cette primitive !
    commandBuffer.bindVertexBuffers(0, {globalVertexBuffer}, {byteOffset});

    // 3. On dessine ! 
    // - vertexOffset devient 0 (car le buffer est déjà décalé sur la primitive)
    // - firstIndex est toujours là pour décaler la lecture dans l'IndexBuffer
    commandBuffer.drawIndexed(indexCount, 1, firstIndex, 0, 0);
}


GltfPrimitive::GltfPrimitive(uint32_t firstIndex, uint32_t indexCount, vk::DeviceSize byteOffset, int materialIndex, vk::VertexInputBindingDescription2EXT binding, std::vector<vk::VertexInputAttributeDescription2EXT> attributes)
    : firstIndex(firstIndex), indexCount(indexCount), byteOffset(byteOffset), materialIndex(materialIndex), vertexBindingDescription(binding), vertexAttributeDescriptions(std::move(attributes)) {}

GltfNode::GltfNode(GltfModel& model, tinygltf::Model& root, tinygltf::Node node, glm::mat4 parent_node_transform) {
    node_transform = glm::mat4(1.0);
    if (node.translation.size() == 3) {
        glm::vec3 translation(node.translation[0], node.translation[1], node.translation[2]);

        node_transform = glm::translate(glm::mat4(1.0), translation);
    }

    if (node.rotation.size() == 4) {
        glm::quat rotation(node.rotation[3], node.rotation[0], node.rotation[1], node.rotation[2]);
        glm::mat4 rotation_mat = glm::mat4_cast(rotation);

        node_transform *= rotation_mat;
    }

    if (node.scale.size() == 3) {
        glm::vec3 scale(node.scale[0], node.scale[1], node.scale[2]);

        node_transform = glm::scale(node_transform, scale);
    }

    node_transform = parent_node_transform * node_transform;
    
    if (node.mesh >= 0 && static_cast<size_t>(node.mesh) < root.meshes.size()) {
        mesh = &(model.meshes[node.mesh]);
    } else {
        mesh = nullptr;
    }


    for (size_t i = 0; i < node.children.size(); i++) {
        assert((node.children[i] >= 0) && (static_cast<size_t>(node.children[i]) < root.nodes.size()));
        children.push_back(GltfNode(model, root, root.nodes[node.children[i]], node_transform));
    }
}

void GltfModel::createVertexBuffer()
{
    if (globalVertexData.empty()) return;
    vk::DeviceSize bufferSize = globalVertexData.size();

    auto [stagingBuffer, stagingBufferMemory] =
        VulkanUtils::createBuffer(*device, *physicalDevice, bufferSize, vk::BufferUsageFlagBits::eTransferSrc,
                     vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);

    vk::MemoryRequirements memRequirementsStaging = stagingBuffer.getMemoryRequirements();

    void *dataStaging = stagingBufferMemory.mapMemory(0, bufferSize);
    memcpy(dataStaging, globalVertexData.data(), bufferSize);
    stagingBufferMemory.unmapMemory();

    std::tie(globalVertexBuffer, globalVertexMemory) =
        VulkanUtils::createBuffer(*device, *physicalDevice, bufferSize, vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferDst,
                     vk::MemoryPropertyFlagBits::eDeviceLocal);

    VulkanUtils::copyBuffer(*device, *commandPool, *graphicsQueue, stagingBuffer, globalVertexBuffer, bufferSize);
}

void GltfModel::createIndexBuffer()
{
    if (indices.empty()) return;
    vk::DeviceSize bufferSize = sizeof(indices[0]) * indices.size();

    auto [stagingBuffer, stagingBufferMemory] =
        VulkanUtils::createBuffer(*device, *physicalDevice, bufferSize, vk::BufferUsageFlagBits::eTransferSrc,
                     vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);

    void *data = stagingBufferMemory.mapMemory(0, bufferSize);
    memcpy(data, indices.data(), (size_t)bufferSize);
    stagingBufferMemory.unmapMemory();

    std::tie(globalIndexBuffer, globalIndexMemory) =
        VulkanUtils::createBuffer(*device, *physicalDevice, bufferSize, vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eTransferDst,
                     vk::MemoryPropertyFlagBits::eDeviceLocal);

    VulkanUtils::copyBuffer(*device, *commandPool, *graphicsQueue, stagingBuffer, globalIndexBuffer, bufferSize);
}

GltfModel::GltfModel(const std::string& path, vk::raii::Device& device, vk::raii::PhysicalDevice& physicalDevice, vk::raii::CommandPool& commandPool, vk::raii::Queue& graphicsQueue)
    : device(&device), physicalDevice(&physicalDevice), commandPool(&commandPool), graphicsQueue(&graphicsQueue)
{
    // Use tinygltf to load the model instead of tinyobjloader
    tinygltf::Model    model;
    tinygltf::TinyGLTF loader;
    std::string        err;
    std::string        warn;

    bool ret = loader.LoadBinaryFromFile(&model, &err, &warn, path);

    if (!warn.empty())
    {
        std::cout << "glTF warning: " << warn << std::endl;
    }

    if (!err.empty())
    {
        std::cout << "glTF error: " << err << std::endl;
    }

    if (!ret)
    {
        throw std::runtime_error("Failed to load glTF model");
    }

    // Process all meshes in the model
    for (const auto &mesh : model.meshes)
    {
        meshes.emplace_back();
        for (const auto &primitive : mesh.primitives)
        {
            GltfMesh &gltfMesh = meshes.back();

            // Get indices
            const tinygltf::Accessor   &indexAccessor   = model.accessors[primitive.indices];
            const tinygltf::BufferView &indexBufferView = model.bufferViews[indexAccessor.bufferView];
            const tinygltf::Buffer     &indexBuffer     = model.buffers[indexBufferView.buffer];

            // Get vertex positions
            const tinygltf::Accessor   &posAccessor   = model.accessors[primitive.attributes.at("POSITION")];
            const tinygltf::BufferView &posBufferView = model.bufferViews[posAccessor.bufferView];
            const tinygltf::Buffer     &posBuffer     = model.buffers[posBufferView.buffer];

            // Get texture coordinates if available
            bool                        hasTexCoords       = primitive.attributes.find("TEXCOORD_0") != primitive.attributes.end();
            const tinygltf::Accessor   *texCoordAccessor   = nullptr;
            const tinygltf::BufferView *texCoordBufferView = nullptr;
            const tinygltf::Buffer     *texCoordBuffer     = nullptr;

            if (hasTexCoords)
            {
                texCoordAccessor   = &model.accessors[primitive.attributes.at("TEXCOORD_0")];
                texCoordBufferView = &model.bufferViews[texCoordAccessor->bufferView];
                texCoordBuffer     = &model.buffers[texCoordBufferView->buffer];
            }

            size_t posStride = posAccessor.ByteStride(posBufferView) ? posAccessor.ByteStride(posBufferView) : sizeof(float) * 3;
            size_t texStride = 0;
            if (hasTexCoords) {
                texStride = texCoordAccessor->ByteStride(*texCoordBufferView) ? texCoordAccessor->ByteStride(*texCoordBufferView) : sizeof(float) * 2;
            }

            std::vector<vk::VertexInputAttributeDescription2EXT> attributes;
            uint32_t currentOffset = 0;

            attributes.push_back({
                .location = 0,
                .binding = 0,
                .format = vk::Format::eR32G32B32Sfloat,
                .offset = currentOffset
            });
            currentOffset += sizeof(float) * 3;

            attributes.push_back({
                .location = 1,
                .binding = 0,
                .format = vk::Format::eR32G32B32Sfloat,
                .offset = currentOffset
            });
            currentOffset += sizeof(float) * 3;

            if (hasTexCoords) {
                attributes.push_back({
                    .location = 2,
                    .binding = 0,
                    .format = vk::Format::eR32G32Sfloat,
                    .offset = currentOffset
                });
                currentOffset += sizeof(float) * 2;
            }

            vk::VertexInputBindingDescription2EXT binding = {
                .binding = 0,
                .stride = currentOffset,
                .inputRate = vk::VertexInputRate::eVertex,
                .divisor = 1
            };

            vk::DeviceSize byteOffset = globalVertexData.size();

            for (size_t i = 0; i < posAccessor.count; i++)
            {
                const float *pos = reinterpret_cast<const float *>(&posBuffer.data[posBufferView.byteOffset + posAccessor.byteOffset + i * posStride]);
                float p[3] = {pos[0], pos[1], pos[2]};
                unsigned char* pBytes = reinterpret_cast<unsigned char*>(p);
                globalVertexData.insert(globalVertexData.end(), pBytes, pBytes + sizeof(p));

                float c[3] = {1.0f, 1.0f, 1.0f};
                unsigned char* cBytes = reinterpret_cast<unsigned char*>(c);
                globalVertexData.insert(globalVertexData.end(), cBytes, cBytes + sizeof(c));

                if (hasTexCoords) {
                    const float *texCoord = reinterpret_cast<const float *>(&texCoordBuffer->data[texCoordBufferView->byteOffset + texCoordAccessor->byteOffset + i * texStride]);
                    float t[2] = {texCoord[0], texCoord[1]};
                    unsigned char* tBytes = reinterpret_cast<unsigned char*>(t);
                    globalVertexData.insert(globalVertexData.end(), tBytes, tBytes + sizeof(t));
                }
            }

            uint32_t firstIndex = static_cast<uint32_t>(indices.size());

            const unsigned char *indexData   = &indexBuffer.data[indexBufferView.byteOffset + indexAccessor.byteOffset];
            size_t               indexCount  = indexAccessor.count;
            size_t               indexStride = 0;

            // Determine index stride based on component type
            if (indexAccessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT)
            {
                indexStride = sizeof(uint16_t);
            }
            else if (indexAccessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT)
            {
                indexStride = sizeof(uint32_t);
            }
            else if (indexAccessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE)
            {
                indexStride = sizeof(uint8_t);
            }
            else
            {
                throw std::runtime_error("Unsupported index component type");
            }

            indices.reserve(indices.size() + indexCount);

            for (size_t i = 0; i < indexCount; i++)
            {
                uint32_t index = 0;

                if (indexAccessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT)
                {
                    index = *reinterpret_cast<const uint16_t *>(indexData + i * indexStride);
                }
                else if (indexAccessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT)
                {
                    index = *reinterpret_cast<const uint32_t *>(indexData + i * indexStride);
                }
                else if (indexAccessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE)
                {
                    index = *reinterpret_cast<const uint8_t *>(indexData + i * indexStride);
                }

                indices.push_back(index);
            }

            // Create a GltfPrimitive and add it to the GltfMesh
            GltfPrimitive gltfPrimitive(firstIndex, static_cast<uint32_t>(indexCount), byteOffset, primitive.material, binding, attributes);
            gltfMesh.addPrimitive(std::move(gltfPrimitive));
        }
    }

    // Then create the vertex and index buffers
    createVertexBuffer();
    createIndexBuffer();

    // Finally, we can create the root nodes for the scene
    const tinygltf::Scene &scene = model.scenes[model.defaultScene > -1 ? model.defaultScene : 0];

    for (int nodeIndex : scene.nodes) {
        rootNodes.emplace_back(*this, model, model.nodes[nodeIndex], glm::mat4(1.0f));
    }
}