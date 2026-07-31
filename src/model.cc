#include <iostream>
#include <optional>
#include <algorithm>

#include "model.hpp"

#include "vulkan/vulkan.hpp"
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
    std::cout << "[DEBUG] Metallic = " << metallic_factor << ", Roughness = " << roughness_factor << "\n";
    std::cout << "[DEBUG] Has RM texture: " << (metallic_roughness_texinfo.index != -1 ? "yes" : "no") << "\n";
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

void GltfMaterial::bind(vk::raii::CommandBuffer &commandBuffer, vk::raii::PipelineLayout &pipelineLayout, glm::mat4 modelMatrix, glm::mat4 prevModelMatrix) const
{
    MeshPushConstants pushConstants;
    pushConstants.prevModel = prevModelMatrix;
    pushConstants.modelMatrix = modelMatrix;
    pushConstants.albedoTextureIndex = baseColorTextureIndex.value_or(0xFFFFFFFF);
    pushConstants.rmTextureIndex = metallicRoughnessTextureIndex.value_or(0xFFFFFFFF);

    pushConstants.baseColor.x = basecolor[0];
    pushConstants.baseColor.y = basecolor[1];
    pushConstants.baseColor.z = basecolor[2];
    pushConstants.baseColor.w = 1.0f;

    pushConstants.roughnessFactor = roughness_factor;
    pushConstants.metallicFactor = metallic_factor;

    pushConstants.emissiveColor = glm::vec4(getEmissive(), 0.0f);

    pushConstants.activeAttributes = features;

    commandBuffer.pushConstants(pipelineLayout, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, sizeof(MeshPushConstants), &pushConstants);
}

void GltfPrimitive::draw(vk::raii::CommandBuffer &commandBuffer, vk::raii::PipelineLayout &pipelineLayout, vk::raii::Buffer &globalVertexBuffer, glm::mat4 modelMatrix, glm::mat4 prevModelMatrix) const
{
    commandBuffer.setVertexInputEXT(vertexBindingDescription, vertexAttributeDescriptions);

    commandBuffer.bindVertexBuffers(0, {globalVertexBuffer}, {0});

    material.bind(commandBuffer, pipelineLayout, modelMatrix, prevModelMatrix);

    uint32_t vertexOffset = static_cast<uint32_t>(byteOffset / sizeof(Vertex));
    commandBuffer.drawIndexed(indexCount, 1, firstIndex, static_cast<int32_t>(vertexOffset), 0);
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
            .indexType = primitive.getIndexCount() > 0 ? vk::IndexType::eUint32 : vk::IndexType::eNoneKHR,
            .indexData = primitive.getIndexCount() > 0 ? indexBufferAddress + (primitive.getFirstIndex() * sizeof(uint32_t)) : 0,
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
        .flags = vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace | 
            vk::BuildAccelerationStructureFlagBitsKHR::eAllowUpdate,
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

    std::tie(updateScratchBuffer, updateScratchMemory) = VulkanUtils::createBuffer(
        device,
        physicalDevice,
        buildSizes.updateScratchSize,
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress,
        vk::MemoryPropertyFlagBits::eDeviceLocal);

    vk::BufferDeviceAddressInfo updateScratchAddressInfo{.buffer = *updateScratchBuffer};
    updateScratchAddress = device.getBufferAddress(updateScratchAddressInfo);

    vk::raii::CommandBuffer cmd = VulkanUtils::beginSingleTimeCommands(device, commandPool);

    const vk::AccelerationStructureBuildRangeInfoKHR *pBuildRangeInfo = buildRanges.data();
    cmd.buildAccelerationStructuresKHR(buildInfo, pBuildRangeInfo);

    VulkanUtils::endSingleTimeCommands(std::move(cmd), graphicsQueue);
}

void GltfMesh::updateBlas(vk::raii::CommandBuffer& commandBuffer, vk::DeviceAddress vertexBufferAddress, vk::DeviceAddress indexBufferAddress)
{
    if (primitives.empty()) return;

    std::vector<vk::AccelerationStructureGeometryKHR> geometries;
    geometries.reserve(primitives.size());
    std::vector<vk::AccelerationStructureBuildRangeInfoKHR> buildRanges;
    buildRanges.reserve(primitives.size());

    for (const auto &primitive : primitives)
    {
        vk::AccelerationStructureGeometryTrianglesDataKHR triangles{
            .vertexFormat = vk::Format::eR32G32B32Sfloat,
            .vertexData = vertexBufferAddress + primitive.getByteOffset(),
            .vertexStride = primitive.getStride(),
            .maxVertex = primitive.getVertexCount() - 1,
            .indexType = primitive.getIndexCount() > 0 ? vk::IndexType::eUint32 : vk::IndexType::eNoneKHR,
            .indexData = primitive.getIndexCount() > 0 ? indexBufferAddress + (primitive.getFirstIndex() * sizeof(uint32_t)) : 0,
            .transformData = nullptr};

        geometries.push_back(vk::AccelerationStructureGeometryKHR{
            .geometryType = vk::GeometryTypeKHR::eTriangles,
            .geometry = triangles,
            .flags = vk::GeometryFlagBitsKHR::eOpaque});

        buildRanges.push_back(vk::AccelerationStructureBuildRangeInfoKHR{
            .primitiveCount = primitive.getIndexCount() / 3,
            .primitiveOffset = 0,
            .firstVertex = 0,
            .transformOffset = 0});
    }

    vk::AccelerationStructureBuildGeometryInfoKHR buildInfo{
        .type = vk::AccelerationStructureTypeKHR::eBottomLevel,
        .flags = vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace | vk::BuildAccelerationStructureFlagBitsKHR::eAllowUpdate,
        .mode = vk::BuildAccelerationStructureModeKHR::eUpdate,
        .srcAccelerationStructure = *blasHandle,
        .dstAccelerationStructure = *blasHandle,
        .geometryCount = static_cast<uint32_t>(geometries.size()),
        .pGeometries = geometries.data(),
        .scratchData = updateScratchAddress};

    const vk::AccelerationStructureBuildRangeInfoKHR *pBuildRangeInfo = buildRanges.data();
    commandBuffer.buildAccelerationStructuresKHR(buildInfo, pBuildRangeInfo);
}

GltfNode::GltfNode(GltfModel &model, tinygltf::Model &root, int nodeIndex, glm::mat4 parent_node_transform)
{
    gltfNodeIndex = nodeIndex;
    tinygltf::Node& node = root.nodes[nodeIndex];
    node_transform = glm::mat4(1.0);
    if (node.translation.size() == 3)
    {
        translation = glm::vec3(node.translation[0], node.translation[1], node.translation[2]);
    }

    if (node.rotation.size() == 4)
    {
        // glTF rotation is [x, y, z, w]. GLM quat takes (w, x, y, z)
        rotation = glm::quat(node.rotation[3], node.rotation[0], node.rotation[1], node.rotation[2]);
    }

    if (node.scale.size() == 3)
    {
        scale = glm::vec3(node.scale[0], node.scale[1], node.scale[2]);
    }

    if (node.matrix.size() == 16)
    {
        has_matrix = true;
        const double* m = node.matrix.data();
        local_matrix = glm::mat4(
            m[0], m[1], m[2], m[3],
            m[4], m[5], m[6], m[7],
            m[8], m[9], m[10], m[11],
            m[12], m[13], m[14], m[15]
        );
    }

    initialTranslation = translation;
    initialRotation = rotation;
    initialScale = scale;
    initialMatrix = local_matrix;

    updateNodeTransform();

    if (node.mesh >= 0 && static_cast<size_t>(node.mesh) < root.meshes.size())
    {
        mesh = &(model.meshes[node.mesh]);
    }
    else
    {
        mesh = nullptr;
    }

    if (node.children.size() > 0)
    {
        for (size_t i = 0; i < node.children.size(); i++)
        {
            children.push_back(GltfNode(model, root, node.children[i], glm::mat4(1.0)));
        }
    }
}

void GltfModel::updateBlas(vk::raii::CommandBuffer& commandBuffer, vk::raii::Device& device) {
    if (skeletons.empty()) return;
    
    vk::DeviceAddress vAddr = 0;
    if (*outputVertexBuffer) {
        vk::BufferDeviceAddressInfo vInfo{.buffer = *outputVertexBuffer};
        vAddr = device.getBufferAddress(vInfo);
    }
    
    vk::DeviceAddress iAddr = 0;
    if (*globalIndexBuffer) {
        vk::BufferDeviceAddressInfo iInfo{.buffer = *globalIndexBuffer};
        iAddr = device.getBufferAddress(iInfo);
    }

    for (auto& mesh : meshes) {
        mesh.updateBlas(commandBuffer, vAddr, iAddr);
    }
}

void GltfNode::populateTlasInstances(
    std::vector<vk::AccelerationStructureInstanceKHR> &instances,
    std::vector<InstanceData> &instanceData,
    vk::raii::Device &device,
    glm::mat4 parentMatrix,
    uint32_t &customIndexOffset,
    vk::DeviceAddress vAddr,
    vk::DeviceAddress iAddr,
    bool isSkinned,
    glm::mat4 rootTransform) const
{
    glm::mat4 globalTransform = parentMatrix * node_transform;
    if (mesh)
    {
        vk::DeviceAddress blasAddress = mesh->getBlasAddress(device);
        glm::mat4 renderTransform = isSkinned ? rootTransform : globalTransform;
        vk::TransformMatrixKHR tm = VulkanUtils::glmToVkTransformMatrix(renderTransform);

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
        child.populateTlasInstances(instances, instanceData, device, globalTransform, customIndexOffset, vAddr, iAddr, isSkinned, rootTransform);
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

    void *dataStaging = stagingBufferMemory.mapMemory(0, bufferSize);
    memcpy(dataStaging, globalVertexData.data(), bufferSize);
    stagingBufferMemory.unmapMemory();

    std::tie(inputVertexBuffer, inputVertexBufferMemory) =
        VulkanUtils::createBuffer(*device, *physicalDevice, bufferSize, vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eShaderDeviceAddress | vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR,
                                  vk::MemoryPropertyFlagBits::eDeviceLocal);

    VulkanUtils::copyBuffer(*device, *commandPool, *graphicsQueue, stagingBuffer, inputVertexBuffer, bufferSize);
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

void GltfModel::createJoinMatrixBuffer() {
    // outputVertexBuffer
    vk::DeviceSize bufferSize = globalVertexData.size();

    std::tie(outputVertexBuffer, outputVertexBufferMemory) = VulkanUtils::createBuffer(*device, *physicalDevice, bufferSize, vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress | vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR, vk::MemoryPropertyFlagBits::eDeviceLocal);

    // joinMatrixBuffer
    uint32_t totalBones = 0;
    for (const auto& skeleton : skeletons) {
        totalBones += skeleton.joints.size();
    }

    if (totalBones > 0) {
        vk::DeviceSize jointBufferSize = totalBones * sizeof(glm::mat4);

        std::tie(joinMatrixBuffer, joinMatrixBufferMemory) = VulkanUtils::createBuffer(
            *device, 
            *physicalDevice, 
            jointBufferSize, 
            vk::BufferUsageFlagBits::eStorageBuffer,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent
        );
    }
}

GltfModel::GltfModel(const std::string &path, vk::raii::Device &device, vk::raii::PhysicalDevice &physicalDevice, vk::raii::CommandPool &commandPool, vk::raii::Queue &graphicsQueue, TextureManager &textureManager, SkinMgr& skinMgr)
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
    // Pre-calculate mesh-to-skin mapping to correctly offset joint indices
    std::vector<int> meshToSkin(model.meshes.size(), -1);
    for (const auto& node : model.nodes) {
        if (node.mesh >= 0 && node.skin >= 0) {
            meshToSkin[node.mesh] = node.skin;
        }
    }
    
    std::cout << "[DEBUG] Loaded " << model.skins.size() << " skins for model " << path << std::endl;

    for (size_t meshIndex = 0; meshIndex < model.meshes.size(); ++meshIndex)
    {
        const auto& mesh = model.meshes[meshIndex];
        int skinIndex = meshToSkin[meshIndex];
        uint32_t jointOffset = 0;
        if (skinIndex >= 0) {
            for (int s = 0; s < skinIndex; s++) {
                jointOffset += model.skins[s].joints.size();
            }
        }

        std::vector<vk::AccelerationStructureGeometryKHR> geometries;

        meshes.emplace_back();
        for (const auto &primitive : mesh.primitives)
        {
            GltfMesh &gltfMesh = meshes.back();

            // Get indices
            const tinygltf::Accessor *indexAccessorPtr = nullptr;
            const tinygltf::BufferView *indexBufferViewPtr = nullptr;
            const tinygltf::Buffer *indexBufferPtr = nullptr;
            
            if (primitive.indices > -1) {
                indexAccessorPtr = &model.accessors[primitive.indices];
                indexBufferViewPtr = &model.bufferViews[indexAccessorPtr->bufferView];
                indexBufferPtr = &model.buffers[indexBufferViewPtr->buffer];
            }

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

            // Get joints if available
            bool hasJoints = primitive.attributes.find("JOINTS_0") != primitive.attributes.end();
            const tinygltf::Accessor *jointAccessor = nullptr;
            const tinygltf::BufferView *jointBufferView = nullptr;
            const tinygltf::Buffer *jointBuffer = nullptr;

            if (hasJoints)
            {
                jointAccessor = &model.accessors[primitive.attributes.at("JOINTS_0")];
                jointBufferView = &model.bufferViews[jointAccessor->bufferView];
                jointBuffer = &model.buffers[jointBufferView->buffer];
            }

            // Get weights if available
            bool hasWeights = primitive.attributes.find("WEIGHTS_0") != primitive.attributes.end();
            const tinygltf::Accessor *weightAccessor = nullptr;
            const tinygltf::BufferView *weightBufferView = nullptr;
            const tinygltf::Buffer *weightBuffer = nullptr;

            if (hasWeights)
            {
                weightAccessor = &model.accessors[primitive.attributes.at("WEIGHTS_0")];
                weightBufferView = &model.bufferViews[weightAccessor->bufferView];
                weightBuffer = &model.buffers[weightBufferView->buffer];
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
            size_t jointStride = 0;
            if (hasJoints)
            {
                jointStride = jointAccessor->ByteStride(*jointBufferView) ? jointAccessor->ByteStride(*jointBufferView) : (jointAccessor->componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT ? sizeof(uint16_t) * 4 : sizeof(uint8_t) * 4);
            }
            size_t weightStride = 0;
            if (hasWeights)
            {
                weightStride = weightAccessor->ByteStride(*weightBufferView) ? weightAccessor->ByteStride(*weightBufferView) : sizeof(float) * 4;
            }

            // Calculate the vertex input attribute descriptions and binding description
            std::vector<vk::VertexInputAttributeDescription2EXT> attributes;

            attributes.push_back({.location = 0,
                                  .binding = 0,
                                  .format = vk::Format::eR32G32B32A32Sfloat,
                                  .offset = static_cast<uint32_t>(offsetof(Vertex, pos))});

            attributes.push_back({.location = 1,
                                  .binding = 0,
                                  .format = vk::Format::eR32G32B32A32Sfloat,
                                  .offset = static_cast<uint32_t>(offsetof(Vertex, normal))});

            attributes.push_back({.location = 2,
                                  .binding = 0,
                                  .format = vk::Format::eR32G32B32A32Sfloat,
                                  .offset = static_cast<uint32_t>(offsetof(Vertex, uv))});

            attributes.push_back({.location = 3,
                                  .binding = 0,
                                  .format = vk::Format::eR32G32B32A32Uint,
                                  .offset = static_cast<uint32_t>(offsetof(Vertex, joints))});

            attributes.push_back({.location = 4,
                                  .binding = 0,
                                  .format = vk::Format::eR32G32B32A32Sfloat,
                                  .offset = static_cast<uint32_t>(offsetof(Vertex, weights))});

            vk::VertexInputBindingDescription2EXT binding = {
                .binding = 0,
                .stride = sizeof(Vertex),
                .inputRate = vk::VertexInputRate::eVertex,
                .divisor = 1};

            vk::DeviceSize byteOffset = globalVertexData.size();

            for (size_t i = 0; i < posAccessor.count; i++)
            {
                Vertex vertex{};

                const unsigned char *posPtr = &posBuffer.data[posBufferView.byteOffset + posAccessor.byteOffset + i * posStride];
                if (posAccessor.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT) {
                    const float *pos = reinterpret_cast<const float *>(posPtr);
                    vertex.pos = glm::vec4(pos[0], pos[1], pos[2], 1.0f);
                } else if (posAccessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT) {
                    const uint16_t *pos = reinterpret_cast<const uint16_t *>(posPtr);
                    vertex.pos = glm::vec4(pos[0], pos[1], pos[2], 1.0f);
                } else if (posAccessor.componentType == TINYGLTF_COMPONENT_TYPE_SHORT) {
                    const int16_t *pos = reinterpret_cast<const int16_t *>(posPtr);
                    vertex.pos = glm::vec4(pos[0], pos[1], pos[2], 1.0f);
                } else {
                    vertex.pos = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
                }

                if (hasNormals)
                {
                    const unsigned char *normalPtr = &normalBuffer->data[normalBufferView->byteOffset + normalAccessor->byteOffset + i * normalStride];
                    if (normalAccessor->componentType == TINYGLTF_COMPONENT_TYPE_FLOAT) {
                        const float *normal = reinterpret_cast<const float *>(normalPtr);
                        vertex.normal = glm::vec4(normal[0], normal[1], normal[2], 0.0f);
                    } else if (normalAccessor->componentType == TINYGLTF_COMPONENT_TYPE_SHORT) {
                        const int16_t *normal = reinterpret_cast<const int16_t *>(normalPtr);
                        vertex.normal = glm::vec4(normal[0] / 32767.0f, normal[1] / 32767.0f, normal[2] / 32767.0f, 0.0f);
                    } else if (normalAccessor->componentType == TINYGLTF_COMPONENT_TYPE_BYTE) {
                        const int8_t *normal = reinterpret_cast<const int8_t *>(normalPtr);
                        vertex.normal = glm::vec4(normal[0] / 127.0f, normal[1] / 127.0f, normal[2] / 127.0f, 0.0f);
                    } else {
                        vertex.normal = glm::vec4(0.0f, 1.0f, 0.0f, 0.0f);
                    }
                } else {
                    vertex.normal = glm::vec4(0.0f, 1.0f, 0.0f, 0.0f);
                }

                if (hasTexCoords)
                {
                    const unsigned char *tcPtr = &texCoordBuffer->data[texCoordBufferView->byteOffset + texCoordAccessor->byteOffset + i * texStride];

                    if (texCoordAccessor->componentType == TINYGLTF_COMPONENT_TYPE_FLOAT) {
                        const float *texCoord = reinterpret_cast<const float *>(tcPtr);
                        vertex.uv = glm::vec4(texCoord[0], texCoord[1], 0.0f, 0.0f);
                    } else if (texCoordAccessor->componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT) {
                        const uint16_t *texCoord = reinterpret_cast<const uint16_t *>(tcPtr);
                        vertex.uv = glm::vec4(texCoord[0] / 65535.0f, texCoord[1] / 65535.0f, 0.0f, 0.0f);
                    } else if (texCoordAccessor->componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE) {
                        const uint8_t *texCoord = reinterpret_cast<const uint8_t *>(tcPtr);
                        vertex.uv = glm::vec4(texCoord[0] / 255.0f, texCoord[1] / 255.0f, 0.0f, 0.0f);
                    }
                } else {
                    vertex.uv = glm::vec4(0.0f, 0.0f, 0.0f, 0.0f);
                }

                if (hasJoints)
                {
                    const unsigned char *jointPtr = &jointBuffer->data[jointBufferView->byteOffset + jointAccessor->byteOffset + i * jointStride];
                    if (jointAccessor->componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT) {
                        const uint16_t *joints = reinterpret_cast<const uint16_t *>(jointPtr);
                        vertex.joints = glm::uvec4(joints[0], joints[1], joints[2], joints[3]);
                    } else if (jointAccessor->componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE) {
                        const uint8_t *joints = reinterpret_cast<const uint8_t *>(jointPtr);
                        vertex.joints = glm::uvec4(joints[0], joints[1], joints[2], joints[3]);
                    }
                    vertex.joints += glm::uvec4(jointOffset);
                } else {
                    vertex.joints = glm::uvec4(0, 0, 0, 0);
                }

                if (hasWeights)
                {
                    const unsigned char *weightPtr = &weightBuffer->data[weightBufferView->byteOffset + weightAccessor->byteOffset + i * weightStride];
                    if (weightAccessor->componentType == TINYGLTF_COMPONENT_TYPE_FLOAT) {
                        const float *weights = reinterpret_cast<const float *>(weightPtr);
                        vertex.weights = glm::vec4(weights[0], weights[1], weights[2], weights[3]);
                    } else if (weightAccessor->componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE) {
                        const uint8_t *weights = reinterpret_cast<const uint8_t *>(weightPtr);
                        vertex.weights = glm::vec4(weights[0] / 255.0f, weights[1] / 255.0f, weights[2] / 255.0f, weights[3] / 255.0f);
                    } else if (weightAccessor->componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT) {
                        const uint16_t *weights = reinterpret_cast<const uint16_t *>(weightPtr);
                        vertex.weights = glm::vec4(weights[0] / 65535.0f, weights[1] / 65535.0f, weights[2] / 65535.0f, weights[3] / 65535.0f);
                    }
                } else {
                    vertex.weights = glm::vec4(0.0f, 0.0f, 0.0f, 0.0f);
                }

                unsigned char *vBytes = reinterpret_cast<unsigned char *>(&vertex);
                globalVertexData.insert(globalVertexData.end(), vBytes, vBytes + sizeof(Vertex));
            }

            uint32_t firstIndex = static_cast<uint32_t>(indices.size());
            size_t indexCount = 0;

            if (primitive.indices > -1) {
                const unsigned char *indexData = &indexBufferPtr->data[indexBufferViewPtr->byteOffset + indexAccessorPtr->byteOffset];
                indexCount = indexAccessorPtr->count;
                size_t indexStride = 0;

                // Determine index stride based on component type
                if (indexAccessorPtr->ByteStride(*indexBufferViewPtr) > 0) {
                    indexStride = indexAccessorPtr->ByteStride(*indexBufferViewPtr);
                } else if (indexAccessorPtr->componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT)
                {
                    indexStride = sizeof(uint16_t);
                }
                else if (indexAccessorPtr->componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT)
                {
                    indexStride = sizeof(uint32_t);
                }
                else if (indexAccessorPtr->componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE)
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

                    if (indexAccessorPtr->componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT)
                    {
                        index = *reinterpret_cast<const uint16_t *>(indexData + i * indexStride);
                    }
                    else if (indexAccessorPtr->componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT)
                    {
                        index = *reinterpret_cast<const uint32_t *>(indexData + i * indexStride);
                    }
                    else if (indexAccessorPtr->componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE)
                    {
                        index = *reinterpret_cast<const uint8_t *>(indexData + i * indexStride);
                    }

                    indices.push_back(index);
                }
                
                // Debug print min and max index
                if (!indices.empty()) {
                    uint32_t minIndex = indices[firstIndex];
                    uint32_t maxIndex = indices[firstIndex];
                    for (size_t i = firstIndex; i < indices.size(); i++) {
                        if (indices[i] < minIndex) minIndex = indices[i];
                        if (indices[i] > maxIndex) maxIndex = indices[i];
                    }
                    if (maxIndex >= posAccessor.count) {
                        std::cout << "[ERROR] MAX INDEX OUT OF BOUNDS! The glTF index buffer contains an index that exceeds the vertex count!" << std::endl;
                    }
                }

            } else {
                // If there are no indices, we synthesize them
                indexCount = posAccessor.count;
                indices.reserve(indices.size() + indexCount);
                for (size_t i = 0; i < indexCount; i++) {
                    indices.push_back(static_cast<uint32_t>(i));
                }
            }

            // Get the material of this primitive
            tinygltf::Material material;
            if (primitive.material >= 0 && primitive.material < model.materials.size()) {
                material = model.materials[primitive.material];
            }

            // Create a GltfPrimitive and add it to the GltfMesh
            GltfPrimitive gltfPrimitive(model, firstIndex, static_cast<uint32_t>(indexCount), posAccessor.count, byteOffset, material, hasNormals, binding, attributes, textureManager, path);
            gltfMesh.addPrimitive(std::move(gltfPrimitive));
        }
    }

    // Then create the vertex and index buffers
    createVertexBuffer();
    createIndexBuffer();

    vk::DeviceAddress vertexBufferAddress = 0;
    if (*inputVertexBuffer) {
        vk::BufferDeviceAddressInfo vertexAddressInfo{.buffer = *inputVertexBuffer};
        vertexBufferAddress = device.getBufferAddress(vertexAddressInfo);
    }
    
    vk::DeviceAddress indexBufferAddress = 0;
    if (*globalIndexBuffer) {
        vk::BufferDeviceAddressInfo indexAddressInfo{.buffer = *globalIndexBuffer};
        indexBufferAddress = device.getBufferAddress(indexAddressInfo);
    }

    for (auto &gltfMesh : meshes)
    {
        gltfMesh.buildBlas(device, physicalDevice, commandPool, graphicsQueue, vertexBufferAddress, indexBufferAddress);
    }

    // Finally, we can create the root nodes for the scene
    const tinygltf::Scene &scene = model.scenes[model.defaultScene > -1 ? model.defaultScene : 0];

    for (int nodeIndex : scene.nodes)
    {
        rootNodes.emplace_back(*this, model, nodeIndex, glm::mat4(1.0f));
    }

    // Identify all nodes that are children of other nodes
    std::vector<bool> isChild(model.nodes.size(), false);
    for (const auto& node : model.nodes) {
        for (int child : node.children) {
            if (child >= 0 && child < isChild.size()) {
                isChild[child] = true;
            }
        }
    }

    // Load any other root nodes (like uninstantiated armatures) that aren't in the default scene
    for (size_t i = 0; i < model.nodes.size(); i++) {
        if (!isChild[i]) {
            bool inScene = false;
            for (int sceneNode : scene.nodes) {
                if (sceneNode == static_cast<int>(i)) {
                    inScene = true;
                    break;
                }
            }
            if (!inScene) {
                rootNodes.emplace_back(*this, model, static_cast<int>(i), glm::mat4(1.0f));
            }
        }
    }

    linearNodes.resize(model.nodes.size(), nullptr);
    for (auto& rootNode : rootNodes) {
        populateLinearNodes(rootNode);
    }

    // Load skins
    for (const auto& skin : model.skins) {
        Skeleton skeleton;
        for (int jointIndex : skin.joints) {
            skeleton.joints.push_back(linearNodes[jointIndex]);
        }
        
        if (skin.inverseBindMatrices >= 0) {
            const tinygltf::Accessor &accessor = model.accessors[skin.inverseBindMatrices];
            const tinygltf::BufferView &bufferView = model.bufferViews[accessor.bufferView];
            const tinygltf::Buffer &buffer = model.buffers[bufferView.buffer];
            
            size_t stride = accessor.ByteStride(bufferView) ? accessor.ByteStride(bufferView) : sizeof(glm::mat4);
            skeleton.inverseBindMatrices.reserve(accessor.count);
            for (size_t i = 0; i < accessor.count; ++i) {
                glm::mat4 m;
                memcpy(&m, &buffer.data[bufferView.byteOffset + accessor.byteOffset + i * stride], sizeof(glm::mat4));
                skeleton.inverseBindMatrices.push_back(m);
            }
        }
        
        skeletons.push_back(skeleton);
    }

    // Load animations
    for (const auto& anim : model.animations) {
        Animation animation;
        animation.name = anim.name;

        // Samplers
        for (const auto& samp : anim.samplers) {
            AnimationSampler sampler;
            sampler.interpolation = samp.interpolation;

            // Read inputs (time)
            const tinygltf::Accessor& inputAccessor = model.accessors[samp.input];
            const tinygltf::BufferView& inputBufferView = model.bufferViews[inputAccessor.bufferView];
            const tinygltf::Buffer& inputBuffer = model.buffers[inputBufferView.buffer];
            const float* inputData = reinterpret_cast<const float*>(&inputBuffer.data[inputBufferView.byteOffset + inputAccessor.byteOffset]);
            for (size_t i = 0; i < inputAccessor.count; i++) {
                sampler.inputs.push_back(inputData[i]);
                if (inputData[i] < animation.start) animation.start = inputData[i];
                if (inputData[i] > animation.end) animation.end = inputData[i];
            }

            // Read outputs (values)
            const tinygltf::Accessor& outputAccessor = model.accessors[samp.output];
            const tinygltf::BufferView& outputBufferView = model.bufferViews[outputAccessor.bufferView];
            const tinygltf::Buffer& outputBuffer = model.buffers[outputBufferView.buffer];
            const unsigned char* outputData = &outputBuffer.data[outputBufferView.byteOffset + outputAccessor.byteOffset];

            if (outputAccessor.type == TINYGLTF_TYPE_VEC3) {
                size_t stride = outputAccessor.ByteStride(outputBufferView) ? outputAccessor.ByteStride(outputBufferView) : sizeof(float) * 3;
                for (size_t i = 0; i < outputAccessor.count; i++) {
                    const float* data = reinterpret_cast<const float*>(&outputData[i * stride]);
                    sampler.outputs.push_back(glm::vec4(data[0], data[1], data[2], 0.0f));
                }
            } else if (outputAccessor.type == TINYGLTF_TYPE_VEC4) {
                size_t stride = outputAccessor.ByteStride(outputBufferView) ? outputAccessor.ByteStride(outputBufferView) : sizeof(float) * 4;
                for (size_t i = 0; i < outputAccessor.count; i++) {
                    const float* data = reinterpret_cast<const float*>(&outputData[i * stride]);
                    sampler.outputs.push_back(glm::vec4(data[0], data[1], data[2], data[3]));
                }
            }
            animation.samplers.push_back(sampler);
        }

        // Channels
        for (const auto& source : anim.channels) {
            AnimationChannel channel;
            if (source.target_path == "rotation") {
                channel.path = AnimationPathType::ROTATION;
            } else if (source.target_path == "translation") {
                channel.path = AnimationPathType::TRANSLATION;
            } else if (source.target_path == "scale") {
                channel.path = AnimationPathType::SCALE;
            } else {
                continue;
            }
            channel.samplerIndex = source.sampler;
            channel.node = linearNodes[source.target_node];
            animation.channels.push_back(channel);
        }

        animations.push_back(animation);
    }

    createJoinMatrixBuffer();

    createComputeResources(skinMgr);
}

void GltfModel::createComputeResources(SkinMgr& skinMgr) {
    // S'il n'y a pas d'os, on ne fait rien
    if (skeletons.empty()) return;

    std::array<vk::DescriptorPoolSize, 1> poolSizes = {
        vk::DescriptorPoolSize{vk::DescriptorType::eStorageBuffer, 5}
    };

    vk::DescriptorPoolCreateInfo poolInfo{
        .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
        .maxSets = 1,
        .poolSizeCount = static_cast<uint32_t>(poolSizes.size()),
        .pPoolSizes = poolSizes.data()
    };
    computeDescriptorPool = vk::raii::DescriptorPool(*device, poolInfo);

    vk::DescriptorSetAllocateInfo allocInfo{
        .descriptorPool = *computeDescriptorPool,
        .descriptorSetCount = 1,
        .pSetLayouts = &(*skinMgr.getDescriptorSetLayout())
    };
    skin.descriptor_set = std::move(vk::raii::DescriptorSets(*device, allocInfo).front());

    vk::DescriptorBufferInfo inputInfo{
        .buffer = *inputVertexBuffer,
        .offset = 0,
        .range = vk::WholeSize
    };

    vk::DescriptorBufferInfo jointInfo{
        .buffer = *joinMatrixBuffer,
        .offset = 0,
        .range = vk::WholeSize
    };

    vk::DescriptorBufferInfo outputInfo{
        .buffer = *outputVertexBuffer,
        .offset = 0,
        .range = vk::WholeSize
    };

    std::array<vk::WriteDescriptorSet, 3> descriptorWrites = {
        vk::WriteDescriptorSet{
            .dstSet = *skin.descriptor_set,
            .dstBinding = 0,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .pBufferInfo = &inputInfo
        },
        vk::WriteDescriptorSet{
            .dstSet = *skin.descriptor_set,
            .dstBinding = 1,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .pBufferInfo = &outputInfo
        },
        vk::WriteDescriptorSet{
            .dstSet = *skin.descriptor_set,
            .dstBinding = 2,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .pBufferInfo = &jointInfo
        }
    };

    device->updateDescriptorSets(descriptorWrites, nullptr);

    skin.input_vertex_buffer = &inputVertexBuffer;
    skin.joint_matrix_buffer = &joinMatrixBuffer;
    skin.output_vertex_buffer = &outputVertexBuffer;
    skin.vertex_count = static_cast<uint32_t>(globalVertexData.size() / sizeof(Vertex)); 
}

// Utilitary functions to create the physic shape
void GltfNode::collectPhysicsVertices(const std::vector<unsigned char>& globalVertexData, JPH::Array<JPH::Vec3>& outPositions, glm::mat4 parentMatrix) const {
    glm::mat4 globalTransform = parentMatrix * node_transform;

    if (mesh) {
        for (const auto& primitive : mesh->getPrimitives()) {
            size_t vertexCount = primitive.getVertexCount();
            size_t stride = primitive.getStride();
            size_t byteOffset = primitive.getByteOffset();

            for (size_t i = 0; i < vertexCount; i++) {
                const float* pos = reinterpret_cast<const float*>(&globalVertexData[byteOffset + i * stride]);
                glm::vec4 worldPos = globalTransform * glm::vec4(pos[0], pos[1], pos[2], 1.0f);
                outPositions.push_back(JPH::Vec3(worldPos.x, worldPos.y, worldPos.z));
            }
        }
    }

    for (const auto& child : children) {
        child.collectPhysicsVertices(globalVertexData, outPositions, globalTransform);
    }
}

void GltfNode::collectPhysicsMeshData(
    const std::vector<unsigned char>& globalVertexData,
    const std::vector<uint32_t>& globalIndices,
    JPH::VertexList& outVertices,
    JPH::IndexedTriangleList& outTriangles,
    glm::mat4 parentMatrix) const
{
    glm::mat4 globalTransform = parentMatrix * node_transform;

    if (mesh) {
        for (const auto& primitive : mesh->getPrimitives()) {
            uint32_t baseVertexIndex = static_cast<uint32_t>(outVertices.size());

            size_t vertexCount = primitive.getVertexCount();
            size_t stride = primitive.getStride();
            size_t byteOffset = primitive.getByteOffset();

            for (size_t i = 0; i < vertexCount; i++) {
                const float* pos = reinterpret_cast<const float*>(&globalVertexData[byteOffset + i * stride]);
                glm::vec4 worldPos = globalTransform * glm::vec4(pos[0], pos[1], pos[2], 1.0f);
                outVertices.push_back(JPH::Float3(worldPos.x, worldPos.y, worldPos.z));
            }

            uint32_t firstIndex = primitive.getFirstIndex();
            uint32_t indexCount = primitive.getIndexCount();

            for (uint32_t i = 0; i < indexCount; i += 3) {
                uint32_t idx0 = baseVertexIndex + globalIndices[firstIndex + i + 0];
                uint32_t idx1 = baseVertexIndex + globalIndices[firstIndex + i + 1];
                uint32_t idx2 = baseVertexIndex + globalIndices[firstIndex + i + 2];

                outTriangles.push_back(JPH::IndexedTriangle(idx0, idx1, idx2));
            }
        }
    }

    for (const auto& child : children) {
        child.collectPhysicsMeshData(globalVertexData, globalIndices, outVertices, outTriangles, globalTransform);
    }
}

JPH::ShapeSettings* GltfModel::getConvexHull() const {
    JPH::Array<JPH::Vec3> positions;

    // On parcourt la hiérarchie des nœuds du glTF pour appliquer les transformations de nœuds (node_transform)
    for (const auto& rootNode : rootNodes) {
        rootNode.collectPhysicsVertices(globalVertexData, positions, staticTransform);
    }

    if (positions.empty()) {
        throw std::runtime_error("Aucun sommet trouvé pour générer la Convex Hull !");
    }

    // Jolt se charge de calculer l'enveloppe convexe optimale
    return new JPH::ConvexHullShapeSettings(positions);
}

JPH::ShapeSettings* GltfModel::getMeshShape() const {
    JPH::VertexList vertices;
    JPH::IndexedTriangleList triangles;

    for (const auto& rootNode : rootNodes) {
        rootNode.collectPhysicsMeshData(globalVertexData, indices, vertices, triangles, staticTransform);
    }

    if (vertices.empty() || triangles.empty()) {
        throw std::runtime_error("Aucun sommet ou triangle trouvé pour générer le MeshShape !");
    }

    return new JPH::MeshShapeSettings(vertices, triangles);
}

JPH::ShapeSettings* GltfModel::getBoxShape() const {
    JPH::Array<JPH::Vec3> positions;

    for (const auto& rootNode : rootNodes) {
        rootNode.collectPhysicsVertices(globalVertexData, positions, staticTransform);
    }

    if (positions.empty()) {
        throw std::runtime_error("Aucun sommet trouvé pour générer la BoxShape !");
    }

    JPH::Vec3 minPoint = positions[0];
    JPH::Vec3 maxPoint = positions[0];

    for (const auto& p : positions) {
        minPoint = JPH::Vec3::sMin(minPoint, p);
        maxPoint = JPH::Vec3::sMax(maxPoint, p);
    }

    JPH::Vec3 center = 0.5f * (minPoint + maxPoint);
    JPH::Vec3 halfExtents = 0.5f * (maxPoint - minPoint);

    halfExtents = JPH::Vec3::sMax(halfExtents, JPH::Vec3(0.001f, 0.001f, 0.001f));

    JPH::BoxShapeSettings* boxSettings = new JPH::BoxShapeSettings(halfExtents);

    if (center.LengthSq() > 1e-6f) {
        return new JPH::RotatedTranslatedShapeSettings(center, JPH::Quat::sIdentity(), boxSettings);
    }

    return boxSettings;
}

float GltfModel::getAnimationDuration(uint32_t index) const {
    if (index < animations.size()) {
        return std::max(0.0f, animations[index].end - animations[index].start);
    }
    return 0.0f;
}

void GltfModel::updateAnimation(uint32_t index, float time, bool loop) {
    if (index < animations.size()) {
        Animation& animation = animations[index];

        float animDuration = animation.end - animation.start;
        if (animDuration > 0.0f) {
            float localTime;
            if (loop) {
                localTime = fmod(time, animDuration) + animation.start;
            } else {
                localTime = std::clamp(animation.start + time, animation.start, animation.end);
            }

            for (const auto& channel : animation.channels) {
                AnimationSampler& sampler = animation.samplers[channel.samplerIndex];
                
                if (sampler.inputs.empty()) continue;

                if (sampler.inputs.size() == 1) {
                    glm::vec4 v0 = sampler.outputs[0];
                    if (channel.path == AnimationPathType::TRANSLATION) {
                        channel.node->translation = glm::vec3(v0);
                    } else if (channel.path == AnimationPathType::SCALE) {
                        channel.node->scale = glm::vec3(v0);
                    } else if (channel.path == AnimationPathType::ROTATION) {
                        channel.node->rotation = glm::normalize(glm::quat(v0.w, v0.x, v0.y, v0.z));
                    }
                    channel.node->updateNodeTransform();
                    continue;
                }

                size_t keyframeIndex = 0;
                for (size_t i = 0; i < sampler.inputs.size() - 1; i++) {
                    if (localTime >= sampler.inputs[i] && localTime <= sampler.inputs[i + 1]) {
                        keyframeIndex = i;
                        break;
                    }
                }
                
                // Fix boundary case if localTime is out of bounds
                if (localTime <= sampler.inputs.front()) {
                    keyframeIndex = 0;
                    localTime = sampler.inputs.front();
                } else if (localTime >= sampler.inputs.back()) {
                    keyframeIndex = sampler.inputs.size() - 2;
                    localTime = sampler.inputs.back();
                }

                float t0 = sampler.inputs[keyframeIndex];
                float t1 = sampler.inputs[keyframeIndex + 1];
                float factor = 0.0f;
                if (t1 > t0) {
                    factor = std::clamp((localTime - t0) / (t1 - t0), 0.0f, 1.0f);
                }

                glm::vec4 v0 = sampler.outputs[keyframeIndex];
                glm::vec4 v1 = sampler.outputs[keyframeIndex + 1];

                if (channel.path == AnimationPathType::TRANSLATION) {
                    channel.node->translation = glm::mix(glm::vec3(v0), glm::vec3(v1), factor);
                } else if (channel.path == AnimationPathType::SCALE) {
                    channel.node->scale = glm::mix(glm::vec3(v0), glm::vec3(v1), factor);
                } else if (channel.path == AnimationPathType::ROTATION) {
                    glm::quat q0(v0.w, v0.x, v0.y, v0.z);
                    glm::quat q1(v1.w, v1.x, v1.y, v1.z);
                    if (glm::dot(q0, q1) < 0.0f) {
                        q1 = -q1;
                    }
                    channel.node->rotation = glm::normalize(glm::slerp(q0, q1, factor));
                }

                channel.node->updateNodeTransform();
            }
        }
    }

    for (auto& rootNode : rootNodes) {
        rootNode.updateGlobalTransform(glm::mat4(1.0f));
    }

    std::vector<glm::mat4> finalJointMatrices;
    for (const auto& skeleton : skeletons) {
        for (size_t i = 0; i < skeleton.joints.size(); i++) {
            GltfNode* jointNode = skeleton.joints[i];
            glm::mat4 inverseBindMatrix = skeleton.inverseBindMatrices.empty() ? glm::mat4(1.0f) : skeleton.inverseBindMatrices[i];
            glm::mat4 finalMatrix = jointNode->global_transform * inverseBindMatrix;
            finalJointMatrices.push_back(finalMatrix);
        }
    }

    if (!finalJointMatrices.empty() && joinMatrixBufferMemory != nullptr) {
        void* data = joinMatrixBufferMemory.mapMemory(0, finalJointMatrices.size() * sizeof(glm::mat4));
        memcpy(data, finalJointMatrices.data(), finalJointMatrices.size() * sizeof(glm::mat4));
        joinMatrixBufferMemory.unmapMemory();
    }
}

void GltfModel::resetToBindPose() {
    for (auto& rootNode : rootNodes) {
        rootNode.resetToInitial();
        rootNode.updateGlobalTransform(glm::mat4(1.0f));
    }

    std::vector<glm::mat4> finalJointMatrices;
    for (const auto& skeleton : skeletons) {
        for (size_t i = 0; i < skeleton.joints.size(); i++) {
            GltfNode* jointNode = skeleton.joints[i];
            glm::mat4 inverseBindMatrix = skeleton.inverseBindMatrices.empty() ? glm::mat4(1.0f) : skeleton.inverseBindMatrices[i];
            finalJointMatrices.push_back(jointNode->global_transform * inverseBindMatrix);
        }
    }

    if (!finalJointMatrices.empty() && joinMatrixBufferMemory != nullptr) {
        void* data = joinMatrixBufferMemory.mapMemory(0, finalJointMatrices.size() * sizeof(glm::mat4));
        memcpy(data, finalJointMatrices.data(), finalJointMatrices.size() * sizeof(glm::mat4));
        joinMatrixBufferMemory.unmapMemory();
    }
}