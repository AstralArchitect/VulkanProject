#include "text_manager.hpp"

#include "vulkan_utils.hpp"

extern const int MAX_FRAMES_IN_FLIGHT;

void TextureManager::createTextureSampler()
{
    vk::PhysicalDeviceProperties properties = physicalDevice->getProperties();
    vk::SamplerCreateInfo samplerInfo{
        .magFilter = vk::Filter::eLinear,
        .minFilter = vk::Filter::eLinear,
        .mipmapMode = vk::SamplerMipmapMode::eLinear,
        .addressModeU = vk::SamplerAddressMode::eRepeat,
        .addressModeV = vk::SamplerAddressMode::eRepeat,
        .addressModeW = vk::SamplerAddressMode::eRepeat,
        .mipLodBias = 0.0f,
        .anisotropyEnable = vk::True,
        .maxAnisotropy = properties.limits.maxSamplerAnisotropy,
        .compareEnable = vk::False,
        .compareOp = vk::CompareOp::eAlways,
        .minLod = 0.0f,
        .maxLod = vk::LodClampNone};

    textureSampler = vk::raii::Sampler(*device, samplerInfo);
}

void TextureManager::createDescriptorSetLayout()
{
    vk::DescriptorSetLayoutBinding samplerLayoutBinding{.binding = 1,
                                                        .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                                                        .descriptorCount = 100,
                                                        .stageFlags = vk::ShaderStageFlagBits::eFragment,
                                                        .pImmutableSamplers = nullptr};

    std::array<vk::DescriptorSetLayoutBinding, 1> bindings = {samplerLayoutBinding};

    vk::DescriptorBindingFlags bindFlag = vk::DescriptorBindingFlagBits::ePartiallyBound;
    
    vk::DescriptorSetLayoutBindingFlagsCreateInfo bindingFlagsInfo{
        .bindingCount = 1,
        .pBindingFlags = &bindFlag
    };

    vk::DescriptorSetLayoutCreateInfo layoutInfo{.pNext = &bindingFlagsInfo, .bindingCount = static_cast<uint32_t>(bindings.size()), .pBindings = bindings.data()};
    descriptorSetLayout = vk::raii::DescriptorSetLayout(*device, layoutInfo);
}

void TextureManager::createDescriptorPool()
{
    std::array poolSize{
        vk::DescriptorPoolSize(vk::DescriptorType::eCombinedImageSampler, MAX_FRAMES_IN_FLIGHT * 100)};

    vk::DescriptorPoolCreateInfo poolInfo{
        .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
        .maxSets = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT),
        .poolSizeCount = static_cast<uint32_t>(poolSize.size()),
        .pPoolSizes = poolSize.data()};

    descriptorPool = vk::raii::DescriptorPool(*device, poolInfo);
}

void TextureManager::init(vk::raii::Device& device, vk::raii::PhysicalDevice& physicalDevice, vk::raii::CommandPool& commandPool, vk::raii::Queue& graphicsQueue) {
    this->device = &device;
    this->physicalDevice = &physicalDevice;
    this->commandPool = &commandPool;
    this->graphicsQueue = &graphicsQueue;

    createTextureSampler();
    createDescriptorSetLayout();
    createDescriptorPool();

    vk::DescriptorSetAllocateInfo allocInfo{
        .descriptorPool = *descriptorPool,
        .descriptorSetCount = 1,
        .pSetLayouts = &(*descriptorSetLayout)
    };

    auto sets = device.allocateDescriptorSets(allocInfo);
    descriptorSet = std::move(sets[0]);

    createFallbackTexture();
}

uint32_t TextureManager::loadTexture(const tinygltf::Model& model, int textureIndex) {
    auto cacheKey = std::make_pair(&model, textureIndex);
    auto it = gltfTextureCache.find(cacheKey);
    if (it != gltfTextureCache.end()) {
        return it->second;
    }

    Texture texture{
        .image = nullptr,
        .imageMemory = nullptr,
        .imageView = nullptr,
        .mipLevels = 0
    };
    createTextureImage(model, textureIndex, texture);
    texture.imageView = VulkanUtils::createImageView(*device, *texture.image, vk::Format::eR8G8B8A8Srgb, vk::ImageAspectFlagBits::eColor, texture.mipLevels);

    uint32_t newIndex = static_cast<uint32_t>(textures.size());
    textures.push_back(std::move(texture));
    gltfTextureCache[cacheKey] = newIndex;

    updateDescriptorSet(newIndex);

    std::cout << "[TextureManager] Texture glTF index " << textureIndex 
              << " chargee à l'index global : " << newIndex 
              << " (Total textures : " << textures.size() << ")" << std::endl;

    return newIndex;
}

void TextureManager::generateMipmaps(
    vk::raii::CommandBuffer &commandBuffer,
    vk::raii::Image &image,
    vk::Format imageFormat,
    int32_t texWidth,
    int32_t texHeight,
    uint32_t mipLevels)
{
    vk::FormatProperties formatProperties = physicalDevice->getFormatProperties(imageFormat);

    if (!(formatProperties.optimalTilingFeatures & vk::FormatFeatureFlagBits::eSampledImageFilterLinear))
    {
        throw std::runtime_error("texture image format does not support linear blitting!");
    }

    vk::ImageMemoryBarrier barrier = {
        .srcAccessMask = vk::AccessFlagBits::eTransferWrite,
        .dstAccessMask = vk::AccessFlagBits::eTransferRead,
        .oldLayout = vk::ImageLayout::eTransferDstOptimal,
        .newLayout = vk::ImageLayout::eTransferSrcOptimal,
        .srcQueueFamilyIndex = vk::QueueFamilyIgnored,
        .dstQueueFamilyIndex = vk::QueueFamilyIgnored,
        .image = image,
        .subresourceRange = {
            .aspectMask = vk::ImageAspectFlagBits::eColor,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1}};

    int32_t mipWidth = texWidth;
    int32_t mipHeight = texHeight;

    for (uint32_t i = 1; i < mipLevels; i++)
    {
        barrier.subresourceRange.baseMipLevel = i - 1;
        barrier.oldLayout = vk::ImageLayout::eTransferDstOptimal;
        barrier.newLayout = vk::ImageLayout::eTransferSrcOptimal;
        barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
        barrier.dstAccessMask = vk::AccessFlagBits::eTransferRead;

        commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eTransfer, {}, {}, {}, barrier);

        vk::ImageBlit blit = {
            .srcSubresource = {.aspectMask = vk::ImageAspectFlagBits::eColor, .mipLevel = i - 1, .layerCount = 1},
            .srcOffsets = std::array<vk::Offset3D, 2>({{}, {mipWidth, mipHeight, 1}}),
            .dstSubresource = {.aspectMask = vk::ImageAspectFlagBits::eColor, .mipLevel = i, .layerCount = 1},
            .dstOffsets = std::array<vk::Offset3D, 2>({{}, {1 < mipWidth ? mipWidth / 2 : 1, 1 < mipHeight ? mipHeight / 2 : 1, 1}})};

        commandBuffer.blitImage(image, vk::ImageLayout::eTransferSrcOptimal, image, vk::ImageLayout::eTransferDstOptimal, blit, vk::Filter::eLinear);

        barrier.oldLayout = vk::ImageLayout::eTransferSrcOptimal;
        barrier.newLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
        barrier.srcAccessMask = vk::AccessFlagBits::eTransferRead;
        barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;

        commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eFragmentShader, {}, {}, {}, barrier);

        if (1 < mipWidth)
        {
            mipWidth /= 2;
        }
        if (1 < mipHeight)
        {
            mipHeight /= 2;
        }
    }

    barrier.subresourceRange.baseMipLevel = mipLevels - 1;
    barrier.oldLayout = vk::ImageLayout::eTransferDstOptimal;
    barrier.newLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
    barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
    barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;

    commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eFragmentShader, {}, {}, {}, barrier);
}

void TextureManager::createTextureImage(const tinygltf::Model& model, int textureIndex, Texture& texture) {
    const tinygltf::Texture& gltfTexture = model.textures[textureIndex];
    const tinygltf::Image& image = model.images[gltfTexture.source];

    int texWidth = image.width;
    int texHeight = image.height;
    const unsigned char* pixelData = image.image.data();
    vk::DeviceSize imageSize = image.image.size();
    
    // Beaucoup de GPU ne supportent pas le format RGB (3 canaux) sous Vulkan.
    std::vector<unsigned char> rgbaFallback;
    if (image.component == 3)
    {
        imageSize = texWidth * texHeight * 4;
        rgbaFallback.resize(imageSize);
        for (int i = 0; i < texWidth * texHeight; ++i)
        {
            rgbaFallback[i * 4 + 0] = image.image[i * 3 + 0]; // R
            rgbaFallback[i * 4 + 1] = image.image[i * 3 + 1]; // G
            rgbaFallback[i * 4 + 2] = image.image[i * 3 + 2]; // B
            rgbaFallback[i * 4 + 3] = 255;                    // A (Alpha à 100%)
        }
        pixelData = rgbaFallback.data();
    }
    
    auto [stagingBuffer, stagingBufferMemory] = VulkanUtils::createBuffer(
        *device, 
        *physicalDevice, 
        imageSize, 
        vk::BufferUsageFlagBits::eTransferSrc, 
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent
    );

    void *data = stagingBufferMemory.mapMemory(0, imageSize);
    memcpy(data, pixelData, imageSize);
    stagingBufferMemory.unmapMemory();
    
    texture.mipLevels = static_cast<uint32_t>(std::floor(std::log2(std::max(texWidth, texHeight)))) + 1;

    std::tie(texture.image, texture.imageMemory) =
        VulkanUtils::createImage(*device, *physicalDevice, texWidth, texHeight, vk::Format::eR8G8B8A8Srgb, vk::ImageTiling::eOptimal,
                                 vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled,
                                 vk::MemoryPropertyFlagBits::eDeviceLocal, vk::SampleCountFlagBits::e1, texture.mipLevels);

    vk::raii::CommandBuffer commandBuffer = VulkanUtils::beginSingleTimeCommands(*device, *commandPool);
    VulkanUtils::transitionImageLayout(commandBuffer, texture.image, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal, texture.mipLevels);
    VulkanUtils::copyBufferToImage(commandBuffer, stagingBuffer, texture.image, static_cast<uint32_t>(texWidth), static_cast<uint32_t>(texHeight));

    generateMipmaps(commandBuffer, texture.image, vk::Format::eR8G8B8A8Srgb, texWidth, texHeight, texture.mipLevels);
    VulkanUtils::endSingleTimeCommands(std::move(commandBuffer), *graphicsQueue);
}

void TextureManager::updateDescriptorSet(uint32_t textureIndex)
{
    const auto& texture = textures[textureIndex];

    vk::DescriptorImageInfo imageInfo{
        .sampler = *textureSampler,
        .imageView = *texture.imageView,
        .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal
    };

    vk::WriteDescriptorSet descriptorWrite{
        .dstSet = *descriptorSet,
        .dstBinding = 1,
        .dstArrayElement = textureIndex,
        .descriptorCount = 1,
        .descriptorType = vk::DescriptorType::eCombinedImageSampler,
        .pImageInfo = &imageInfo
    };

    device->updateDescriptorSets(descriptorWrite, nullptr);
}

void TextureManager::createFallbackTexture() {
    Texture texture{
        .image = nullptr,
        .imageMemory = nullptr,
        .imageView = nullptr,
        .mipLevels = 1
    };

    // 1. Données pour un pixel blanc 1x1
    unsigned char whitePixel[] = { 255, 255, 255, 255 };

    // 2. Création du Staging Buffer
    auto [stagingBuffer, stagingBufferMemory] = VulkanUtils::createBuffer(
        *device, *physicalDevice, 4,
        vk::BufferUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent
    );

    void* data = stagingBufferMemory.mapMemory(0, 4);
    memcpy(data, whitePixel, 4);
    stagingBufferMemory.unmapMemory();

    // 3. Création de l'image 1x1 sur le GPU
    std::tie(texture.image, texture.imageMemory) = VulkanUtils::createImage(
        *device, *physicalDevice, 1, 1, vk::Format::eR8G8B8A8Srgb, vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled,
        vk::MemoryPropertyFlagBits::eDeviceLocal, vk::SampleCountFlagBits::e1, 1
    );

    // 4. Copie des données
    vk::raii::CommandBuffer commandBuffer = VulkanUtils::beginSingleTimeCommands(*device, *commandPool);
    VulkanUtils::transitionImageLayout(commandBuffer, texture.image, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal, 1);
    VulkanUtils::copyBufferToImage(commandBuffer, stagingBuffer, texture.image, 1, 1);
    VulkanUtils::transitionImageLayout(commandBuffer, texture.image, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal, 1);
    VulkanUtils::endSingleTimeCommands(std::move(commandBuffer), *graphicsQueue);

    // 5. Création de la View
    texture.imageView = VulkanUtils::createImageView(*device, *texture.image, vk::Format::eR8G8B8A8Srgb, vk::ImageAspectFlagBits::eColor, 1);

    // 6. Enregistrement à l'index 0
    textures.push_back(std::move(texture));
    updateDescriptorSet(0);
}
