#pragma once

#if defined(__INTELLISENSE__) || !defined(USE_CPP20_MODULES)
#define VULKAN_HPP_NO_STRUCT_CONSTRUCTORS
#include <vulkan/vulkan_raii.hpp>
#else
#define VULKAN_HPP_NO_STRUCT_CONSTRUCTORS
import vulkan_hpp;
#endif

#include <tiny_gltf.h>

#include <iostream>
#include <vector>

class TextureManager {
    public:
        TextureManager() = default;
        ~TextureManager() = default;

        void init(vk::raii::Device& device, vk::raii::PhysicalDevice& physicalDevice, vk::raii::CommandPool& commandPool, vk::raii::Queue& graphicsQueue);
        uint32_t loadTexture(const tinygltf::Model& texture, int textureIndex, const std::string& modelPath, bool isSRGB = true);

        const vk::raii::DescriptorSetLayout& getDescriptorSetLayout() const { return descriptorSetLayout; }
        const vk::raii::DescriptorSet& getDescriptorSet() const { return descriptorSet; }
    private:
        struct Texture {
            vk::raii::Image image = nullptr;
            vk::raii::DeviceMemory imageMemory = nullptr;
            vk::raii::ImageView imageView = nullptr;
            uint32_t mipLevels = 0;
        };

        vk::raii::Device* device = nullptr;
        vk::raii::PhysicalDevice* physicalDevice = nullptr;
        vk::raii::CommandPool* commandPool = nullptr;
        vk::raii::Queue* graphicsQueue = nullptr;

        vk::raii::Sampler textureSampler = nullptr;

        // Le layout qui décrit le tableau de textures pour le Pipeline Layout
        vk::raii::DescriptorSetLayout descriptorSetLayout = nullptr;

        // Le pool de descripteurs dédié aux textures
        vk::raii::DescriptorPool descriptorPool = nullptr;

        // Le set de descripteurs final qui contient le tableau de toutes nos textures
        vk::raii::DescriptorSet descriptorSet = nullptr;

        std::map<std::pair<std::string, int>, uint32_t> gltfTextureCache;
        std::vector<Texture> textures;

        void createFallbackTexture();

        void createTextureImage(const tinygltf::Model& model, int textureIndex, Texture& texture, vk::Format format);
        void createTextureSampler();

        void createDescriptorSetLayout();
        void createDescriptorPool();

        void updateDescriptorSet(uint32_t textureIndex);

        void generateMipmaps(vk::raii::CommandBuffer& commandBuffer, vk::raii::Image& image, vk::Format imageFormat, int32_t texWidth, int32_t texHeight, uint32_t mipLevels);
};