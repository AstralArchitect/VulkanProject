#include <iostream>
#include <optional>

#include "model.hpp"

#include "vulkan_utils.hpp"
#include "text_manager.hpp"

GltfMaterial::GltfMaterial(const tinygltf::Model &root, tinygltf::Material material, bool hasNormals, TextureManager &textureManager, const std::string& modelPath)
{
    std::copy(material.pbrMetallicRoughness.baseColorFactor.cbegin(), material.pbrMetallicRoughness.baseColorFactor.cbegin() + 3, basecolor);
    metallic_factor = material.pbrMetallicRoughness.metallicFactor;
    roughness_factor = material.pbrMetallicRoughness.roughnessFactor;

    if (material.emissiveFactor.size() == 3) {
        std::copy(material.emissiveFactor.cbegin(), material.emissiveFactor.cend(), emissive_factor);
    } else {
        emissive_factor[0] = emissive_factor[1] = emissive_factor[2] = 0.0;
    }

    transmission_factor = 0.0;

    auto extIt = material.extensions.find("KHR_materials_transmission");
    if (extIt != material.extensions.end()) 
    {
        const tinygltf::Value& extValue = extIt->second;

        if (extValue.Has("transmissionFactor")) 
        {
            transmission_factor = extValue.Get("transmissionFactor").Get<double>();
        }
    }

    tinygltf::TextureInfo basecolor_texinfo = material.pbrMetallicRoughness.baseColorTexture;
    tinygltf::TextureInfo metallic_roughness_texinfo = material.pbrMetallicRoughness.metallicRoughnessTexture;

    features = 0;
    std::cout << "[DEBUG] GltfMaterial '" << material.name << "' loaded. BaseColorFactor = [" << basecolor[0] << ", " << basecolor[1] << ", " << basecolor[2] << "]\n";
    if (basecolor_texinfo.index >= 0)
    {
        features |= 1; // Set bit 0 for base color texture
        baseColorTextureIndex = textureManager.loadTexture(root, basecolor_texinfo.index, modelPath);
        std::cout << "[DEBUG] Base color texture found! Index: " << basecolor_texinfo.index << ", Global Index: " << baseColorTextureIndex.value_or(0) << "\n";
    }
    else
    {
        std::cout << "[DEBUG] No base color texture found for material '" << material.name << "'\n";
        baseColorTextureIndex = std::nullopt; // No texture
    }

    if (metallic_roughness_texinfo.index >= 0)
    {
        features |= 2; // Set bit 1 for metallic roughness texture
        metallicRoughnessTextureIndex = textureManager.loadTexture(root, metallic_roughness_texinfo.index, modelPath, false);
    }
    else
    {
        metallicRoughnessTextureIndex = std::nullopt; // No texture
    }

    if (hasNormals)
    {
        features |= 4; // Set bit 2 for normals
    }
}

void GltfMaterial::bind(vk::raii::CommandBuffer &commandBuffer, vk::raii::PipelineLayout &pipelineLayout, glm::mat4 modelMatrix) const
{
    MeshPushConstants pushConstants;
    pushConstants.modelMatrix = modelMatrix;
    pushConstants.albedoTextureIndex = baseColorTextureIndex.value_or(0); // Default to 0 if no texture
    pushConstants.rmTextureIndex = metallicRoughnessTextureIndex.value_or(0);

    pushConstants.baseColor.x = basecolor[0];
    pushConstants.baseColor.y = basecolor[1];
    pushConstants.baseColor.z = basecolor[2];
    pushConstants.baseColor.w = 1.0f;

    pushConstants.metallicFactor = metallic_factor;
    pushConstants.roughnessFactor = roughness_factor;
    pushConstants.transmissionFactor = transmission_factor;

    pushConstants.emissiveColor = glm::vec4(getEmissive(), 0.0f);

    pushConstants.activeAttributes = features;

    commandBuffer.pushConstants(pipelineLayout, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, sizeof(MeshPushConstants), &pushConstants);
}

void GltfPrimitive::draw(vk::raii::CommandBuffer &commandBuffer, vk::raii::PipelineLayout &pipelineLayout, vk::raii::Buffer &globalVertexBuffer, glm::mat4 modelMatrix) const
{
    commandBuffer.setVertexInputEXT(vertexBindingDescription, vertexAttributeDescriptions);

    commandBuffer.bindVertexBuffers(0, {globalVertexBuffer}, {byteOffset});

    material.bind(commandBuffer, pipelineLayout, modelMatrix);

    commandBuffer.drawIndexed(indexCount, 1, firstIndex, 0, 0);
}

GltfPrimitive::GltfPrimitive(const tinygltf::Model &root, uint32_t primfirstIndex, uint32_t primIndexCount, uint32_t primVertexCount, vk::DeviceSize primByteOffset, tinygltf::Material gltfMaterial, bool hasNormals, vk::VertexInputBindingDescription2EXT binding, std::vector<vk::VertexInputAttributeDescription2EXT> attributes, TextureManager &textureManager, const std::string& modelPath)
{
    firstIndex = primfirstIndex;
    indexCount = primIndexCount;
    byteOffset = primByteOffset;
    vertexCount = primVertexCount;
    vertexBindingDescription = binding;
    vertexAttributeDescriptions = attributes;

    material = GltfMaterial(root, gltfMaterial, hasNormals, textureManager, modelPath);
}

void GltfMesh::buildBlas(
    vk::raii::Device &device,
    vk::raii::PhysicalDevice &physicalDevice,
    vk::raii::CommandPool &commandPool,
    vk::raii::Queue &graphicsQueue,
    vk::DeviceAddress vertexBufferAddress,
    vk::DeviceAddress indexBufferAddress)
{
    if (primitives.empty())
        return;

    std::vector<vk::AccelerationStructureGeometryKHR> geometries;
    geometries.reserve(primitives.size());

    std::vector<uint32_t> maxPrimCounts;
    maxPrimCounts.reserve(primitives.size());

    std::vector<vk::AccelerationStructureBuildRangeInfoKHR> buildRanges;
    buildRanges.reserve(primitives.size());

    for (const auto &primitive : primitives)
    {
        vk::AccelerationStructureGeometryTrianglesDataKHR triangles{
            .vertexFormat = vk::Format::eR32G32B32Sfloat,
            .vertexData = vertexBufferAddress + primitive.getByteOffset(),
            .vertexStride = primitive.getStride(),
            .maxVertex = primitive.getVertexCount() - 1,
            .indexType = vk::IndexType::eUint32,
            .indexData = indexBufferAddress + (primitive.getFirstIndex() * sizeof(uint32_t)),
            .transformData = nullptr};

        geometries.push_back(vk::AccelerationStructureGeometryKHR{
            .geometryType = vk::GeometryTypeKHR::eTriangles,
            .geometry = triangles,
            .flags = vk::GeometryFlagBitsKHR::eOpaque});

        uint32_t triangleCount = primitive.getIndexCount() / 3;
        maxPrimCounts.push_back(triangleCount);

        buildRanges.push_back(vk::AccelerationStructureBuildRangeInfoKHR{
            .primitiveCount = triangleCount,
            .primitiveOffset = 0,
            .firstVertex = 0,
            .transformOffset = 0});
    }

    vk::AccelerationStructureBuildGeometryInfoKHR buildInfo{
        .type = vk::AccelerationStructureTypeKHR::eBottomLevel,
        .flags = vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace,
        .mode = vk::BuildAccelerationStructureModeKHR::eBuild,
        .srcAccelerationStructure = nullptr,
        .dstAccelerationStructure = nullptr,
        .geometryCount = static_cast<uint32_t>(geometries.size()),
        .pGeometries = geometries.data()};

    vk::AccelerationStructureBuildSizesInfoKHR buildSizes = device.getAccelerationStructureBuildSizesKHR(
        vk::AccelerationStructureBuildTypeKHR::eDevice,
        buildInfo,
        maxPrimCounts);

    std::tie(blasBuffer, blasBufferMemory) = VulkanUtils::createBuffer(
        device,
        physicalDevice,
        buildSizes.accelerationStructureSize,
        vk::BufferUsageFlagBits::eAccelerationStructureStorageKHR | vk::BufferUsageFlagBits::eShaderDeviceAddress,
        vk::MemoryPropertyFlagBits::eDeviceLocal);

    vk::AccelerationStructureCreateInfoKHR createInfo{
        .buffer = *blasBuffer,
        .offset = 0,
        .size = buildSizes.accelerationStructureSize,
        .type = vk::AccelerationStructureTypeKHR::eBottomLevel};
    blasHandle = device.createAccelerationStructureKHR(createInfo);

    buildInfo.dstAccelerationStructure = *blasHandle;

    auto [scratchBuffer, scratchMemory] = VulkanUtils::createBuffer(
        device,
        physicalDevice,
        buildSizes.buildScratchSize,
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress,
        vk::MemoryPropertyFlagBits::eDeviceLocal);

    vk::BufferDeviceAddressInfo scratchAddressInfo{.buffer = *scratchBuffer};
    vk::DeviceAddress scratchAddress = device.getBufferAddress(scratchAddressInfo);
    buildInfo.scratchData.deviceAddress = scratchAddress;

    vk::raii::CommandBuffer cmd = VulkanUtils::beginSingleTimeCommands(device, commandPool);

    const vk::AccelerationStructureBuildRangeInfoKHR *pBuildRangeInfo = buildRanges.data();
    cmd.buildAccelerationStructuresKHR(buildInfo, pBuildRangeInfo);

    VulkanUtils::endSingleTimeCommands(std::move(cmd), graphicsQueue);
}

GltfNode::GltfNode(GltfModel &model, tinygltf::Model &root, tinygltf::Node node, glm::mat4 parent_node_transform)
{
    node_transform = glm::mat4(1.0);
    if (node.translation.size() == 3)
    {
        glm::vec3 translation(node.translation[0], node.translation[1], node.translation[2]);

        node_transform = glm::translate(glm::mat4(1.0), translation);
    }

    if (node.rotation.size() == 4)
    {
        glm::quat rotation(node.rotation[3], node.rotation[0], node.rotation[1], node.rotation[2]);
        glm::mat4 rotation_mat = glm::mat4_cast(rotation);

        node_transform *= rotation_mat;
    }

    if (node.scale.size() == 3)
    {
        glm::vec3 scale(node.scale[0], node.scale[1], node.scale[2]);

        node_transform = glm::scale(node_transform, scale);
    }

    node_transform = parent_node_transform * node_transform;

    if (node.mesh >= 0 && static_cast<size_t>(node.mesh) < root.meshes.size())
    {
        mesh = &(model.meshes[node.mesh]);
    }
    else
    {
        mesh = nullptr;
    }

    for (size_t i = 0; i < node.children.size(); i++)
    {
        assert((node.children[i] >= 0) && (static_cast<size_t>(node.children[i]) < root.nodes.size()));
        children.push_back(GltfNode(model, root, root.nodes[node.children[i]], node_transform));
    }
}

void GltfNode::populateTlasInstances(
    std::vector<vk::AccelerationStructureInstanceKHR> &instances,
    std::vector<InstanceData> &instanceData,
    vk::raii::Device &device,
    glm::mat4 parentMatrix,
    uint32_t &customIndexOffset,
    vk::DeviceAddress vAddr,
    vk::DeviceAddress iAddr) const
{
    glm::mat4 globalTransform = parentMatrix * node_transform;
    if (mesh)
    {
        vk::DeviceAddress blasAddress = mesh->getBlasAddress(device);
        vk::TransformMatrixKHR tm = VulkanUtils::glmToVkTransformMatrix(globalTransform);

        vk::AccelerationStructureInstanceKHR instance{
            .transform = tm,
            .instanceCustomIndex = customIndexOffset,
            .mask = 0xFF,
            .instanceShaderBindingTableRecordOffset = 0,
            .flags = static_cast<VkGeometryInstanceFlagsKHR>(vk::GeometryInstanceFlagBitsKHR::eTriangleFacingCullDisable),
            .accelerationStructureReference = blasAddress};
        instances.push_back(instance);

        for (const auto& prim : mesh->getPrimitives()) {
            InstanceData data{
                .vertexBufferAddress = vAddr + prim.getByteOffset(),
                .indexBufferAddress = iAddr + prim.getFirstIndex() * sizeof(uint32_t),
                .baseColor = prim.getMaterial().getBaseColor(),
                .emissiveColor = glm::vec4(prim.getMaterial().getEmissive(), 1.0f),
                .metallic = prim.getMaterial().getMetallic(),
                .roughness = prim.getMaterial().getRoughness(),
                .transmission = prim.getMaterial().getTransmission(),
                .materialID = prim.getMaterial().getMaterialIndex(),
                .activeAttributes = prim.getMaterial().getActiveAttributes(),
                .vertexStrideWords = prim.getStride() / 4,
                .uvOffsetWords = prim.getUvOffset() / 4,
                .normalOffsetWords = prim.getNormalOffset() / 4 
            };
            instanceData.push_back(data);
        }

        customIndexOffset += mesh->getPrimitives().size();
    }

    for (auto &child : children)
    {
        child.populateTlasInstances(instances, instanceData, device, globalTransform, customIndexOffset, vAddr, iAddr);
    }
}

void GltfModel::createVertexBuffer()
{
    if (globalVertexData.empty())
        return;
    vk::DeviceSize bufferSize = globalVertexData.size();

    auto [stagingBuffer, stagingBufferMemory] =
        VulkanUtils::createBuffer(*device, *physicalDevice, bufferSize, vk::BufferUsageFlagBits::eTransferSrc,
                                  vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);

    vk::MemoryRequirements memRequirementsStaging = stagingBuffer.getMemoryRequirements();

    void *dataStaging = stagingBufferMemory.mapMemory(0, bufferSize);
    memcpy(dataStaging, globalVertexData.data(), bufferSize);
    stagingBufferMemory.unmapMemory();

    std::tie(globalVertexBuffer, globalVertexMemory) =
        VulkanUtils::createBuffer(*device, *physicalDevice, bufferSize, vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eShaderDeviceAddress | vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR,
                                  vk::MemoryPropertyFlagBits::eDeviceLocal);

    VulkanUtils::copyBuffer(*device, *commandPool, *graphicsQueue, stagingBuffer, globalVertexBuffer, bufferSize);
}

void GltfModel::createIndexBuffer()
{
    if (indices.empty())
        return;
    vk::DeviceSize bufferSize = sizeof(indices[0]) * indices.size();

    auto [stagingBuffer, stagingBufferMemory] =
        VulkanUtils::createBuffer(*device, *physicalDevice, bufferSize, vk::BufferUsageFlagBits::eTransferSrc,
                                  vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);

    void *data = stagingBufferMemory.mapMemory(0, bufferSize);
    memcpy(data, indices.data(), (size_t)bufferSize);
    stagingBufferMemory.unmapMemory();

    std::tie(globalIndexBuffer, globalIndexMemory) =
        VulkanUtils::createBuffer(*device, *physicalDevice, bufferSize, vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eShaderDeviceAddress | vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR,
                                  vk::MemoryPropertyFlagBits::eDeviceLocal);

    VulkanUtils::copyBuffer(*device, *commandPool, *graphicsQueue, stagingBuffer, globalIndexBuffer, bufferSize);
}

GltfModel::GltfModel(const std::string &path, vk::raii::Device &device, vk::raii::PhysicalDevice &physicalDevice, vk::raii::CommandPool &commandPool, vk::raii::Queue &graphicsQueue, TextureManager &textureManager)
    : device(&device), physicalDevice(&physicalDevice), commandPool(&commandPool), graphicsQueue(&graphicsQueue)
{
    tinygltf::Model model;
    tinygltf::TinyGLTF loader;
    std::string err;
    std::string warn;

    std::cout << "[DEBUG] --- Loading glTF model from path: " << path << " ---\n";
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
        std::vector<vk::AccelerationStructureGeometryKHR> geometries;

        meshes.emplace_back();
        for (const auto &primitive : mesh.primitives)
        {
            GltfMesh &gltfMesh = meshes.back();

            // Get indices
            const tinygltf::Accessor &indexAccessor = model.accessors[primitive.indices];
            const tinygltf::BufferView &indexBufferView = model.bufferViews[indexAccessor.bufferView];
            const tinygltf::Buffer &indexBuffer = model.buffers[indexBufferView.buffer];

            // Get vertex positions
            const tinygltf::Accessor &posAccessor = model.accessors[primitive.attributes.at("POSITION")];
            const tinygltf::BufferView &posBufferView = model.bufferViews[posAccessor.bufferView];
            const tinygltf::Buffer &posBuffer = model.buffers[posBufferView.buffer];

            // Get texture coordinates if available
            bool hasTexCoords = primitive.attributes.find("TEXCOORD_0") != primitive.attributes.end();
            const tinygltf::Accessor *texCoordAccessor = nullptr;
            const tinygltf::BufferView *texCoordBufferView = nullptr;
            const tinygltf::Buffer *texCoordBuffer = nullptr;

            std::cout << "[DEBUG] Primitive hasTexCoords: " << (hasTexCoords ? "true" : "false") << "\n";

            if (hasTexCoords)
            {
                texCoordAccessor = &model.accessors[primitive.attributes.at("TEXCOORD_0")];
                texCoordBufferView = &model.bufferViews[texCoordAccessor->bufferView];
                texCoordBuffer = &model.buffers[texCoordBufferView->buffer];
                std::cout << "[DEBUG] TEXCOORD_0 componentType: " << texCoordAccessor->componentType << "\n";
            }

            // Get normals if available
            bool hasNormals = primitive.attributes.find("NORMAL") != primitive.attributes.end();
            const tinygltf::Accessor *normalAccessor = nullptr;
            const tinygltf::BufferView *normalBufferView = nullptr;
            const tinygltf::Buffer *normalBuffer = nullptr;

            if (hasNormals)
            {
                normalAccessor = &model.accessors[primitive.attributes.at("NORMAL")];
                normalBufferView = &model.bufferViews[normalAccessor->bufferView];
                normalBuffer = &model.buffers[normalBufferView->buffer];
            }

            size_t posStride = posAccessor.ByteStride(posBufferView) ? posAccessor.ByteStride(posBufferView) : sizeof(float) * 3;
            size_t normalStride = 0;
            if (hasNormals)
            {
                normalStride = normalAccessor->ByteStride(*normalBufferView) ? normalAccessor->ByteStride(*normalBufferView) : sizeof(float) * 3;
            }
            size_t texStride = 0;
            if (hasTexCoords)
            {
                texStride = texCoordAccessor->ByteStride(*texCoordBufferView) ? texCoordAccessor->ByteStride(*texCoordBufferView) : sizeof(float) * 2;
            }

            // Calculate the vertex input attribute descriptions and binding description
            std::vector<vk::VertexInputAttributeDescription2EXT> attributes;
            uint32_t currentOffset = 0;

            attributes.push_back({.location = 0,
                                  .binding = 0,
                                  .format = vk::Format::eR32G32B32Sfloat,
                                  .offset = currentOffset});
            currentOffset += sizeof(float) * 3;

            if (hasNormals)
            {
                attributes.push_back({.location = 1,
                                      .binding = 0,
                                      .format = vk::Format::eR32G32B32Sfloat,
                                      .offset = currentOffset});
                currentOffset += sizeof(float) * 3;
            }

            if (hasTexCoords)
            {
                attributes.push_back({.location = 2,
                                      .binding = 0,
                                      .format = vk::Format::eR32G32Sfloat,
                                      .offset = currentOffset});
                currentOffset += sizeof(float) * 2;
            }

            vk::VertexInputBindingDescription2EXT binding = {
                .binding = 0,
                .stride = currentOffset,
                .inputRate = vk::VertexInputRate::eVertex,
                .divisor = 1};

            vk::DeviceSize byteOffset = globalVertexData.size();

            for (size_t i = 0; i < posAccessor.count; i++)
            {
                const float *pos = reinterpret_cast<const float *>(&posBuffer.data[posBufferView.byteOffset + posAccessor.byteOffset + i * posStride]);
                float p[3] = {pos[0], pos[1], pos[2]};
                unsigned char *pBytes = reinterpret_cast<unsigned char *>(p);
                globalVertexData.insert(globalVertexData.end(), pBytes, pBytes + sizeof(p));

                if (hasNormals)
                {
                    const float *normal = reinterpret_cast<const float *>(&normalBuffer->data[normalBufferView->byteOffset + normalAccessor->byteOffset + i * normalStride]);
                    float n[3] = {normal[0], normal[1], normal[2]};
                    unsigned char *nBytes = reinterpret_cast<unsigned char *>(n);
                    globalVertexData.insert(globalVertexData.end(), nBytes, nBytes + sizeof(n));
                }

                if (hasTexCoords)
                {
                    float t[2] = {0.0f, 0.0f};
                    const unsigned char *tcPtr = &texCoordBuffer->data[texCoordBufferView->byteOffset + texCoordAccessor->byteOffset + i * texStride];

                    if (texCoordAccessor->componentType == TINYGLTF_COMPONENT_TYPE_FLOAT) {
                        const float *texCoord = reinterpret_cast<const float *>(tcPtr);
                        t[0] = texCoord[0];
                        t[1] = texCoord[1];
                    } else if (texCoordAccessor->componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT) {
                        const uint16_t *texCoord = reinterpret_cast<const uint16_t *>(tcPtr);
                        t[0] = texCoord[0] / 65535.0f;
                        t[1] = texCoord[1] / 65535.0f;
                    } else if (texCoordAccessor->componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE) {
                        const uint8_t *texCoord = reinterpret_cast<const uint8_t *>(tcPtr);
                        t[0] = texCoord[0] / 255.0f;
                        t[1] = texCoord[1] / 255.0f;
                    }

                    if (i < 3) {
                        std::cout << "[DEBUG] Vertex " << i << " UV: (" << t[0] << ", " << t[1] << ")\n";
                    }

                    unsigned char *tBytes = reinterpret_cast<unsigned char *>(t);
                    globalVertexData.insert(globalVertexData.end(), tBytes, tBytes + sizeof(t));
                }
            }

            uint32_t firstIndex = static_cast<uint32_t>(indices.size());

            const unsigned char *indexData = &indexBuffer.data[indexBufferView.byteOffset + indexAccessor.byteOffset];
            size_t indexCount = indexAccessor.count;
            size_t indexStride = 0;

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

            // Get the material of this primitive
            tinygltf::Material material = model.materials[primitive.material];

            // Create a GltfPrimitive and add it to the GltfMesh
            GltfPrimitive gltfPrimitive(model, firstIndex, static_cast<uint32_t>(indexCount), posAccessor.count, byteOffset, material, hasNormals, binding, attributes, textureManager, path);
            gltfMesh.addPrimitive(std::move(gltfPrimitive));
        }
    }

    // Then create the vertex and index buffers
    createVertexBuffer();
    createIndexBuffer();

    vk::BufferDeviceAddressInfo vertexAddressInfo{.buffer = *globalVertexBuffer};
    vk::BufferDeviceAddressInfo indexAddressInfo{.buffer = *globalIndexBuffer};
    vk::DeviceAddress vertexBufferAddress = device.getBufferAddress(vertexAddressInfo);
    vk::DeviceAddress indexBufferAddress = device.getBufferAddress(indexAddressInfo);

    for (auto &gltfMesh : meshes)
    {
        gltfMesh.buildBlas(device, physicalDevice, commandPool, graphicsQueue, vertexBufferAddress, indexBufferAddress);
    }

    // Finally, we can create the root nodes for the scene
    const tinygltf::Scene &scene = model.scenes[model.defaultScene > -1 ? model.defaultScene : 0];

    for (int nodeIndex : scene.nodes)
    {
        rootNodes.emplace_back(*this, model, model.nodes[nodeIndex], glm::mat4(1.0f));
    }
}