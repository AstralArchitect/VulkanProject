#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/hash.hpp>

struct UniformBufferObject
{
    glm::mat4 model;
    glm::mat4 view;
    glm::mat4 proj;
};

struct MeshPushConstants {
    glm::mat4 modelMatrix;
    glm::mat4 prevModel;
    uint32_t albedoTextureIndex;
    uint32_t rmTextureIndex;
    alignas(16) glm::vec4 baseColor;
    alignas(16) glm::vec4 emissiveColor;
    float metallicFactor;
    float roughnessFactor;
    float transmissionFactor;
    uint32_t activeAttributes;
};

struct alignas(16) InstanceData {
    uint64_t vertexBufferAddress;
    uint64_t indexBufferAddress;
    glm::vec4 baseColor;
    alignas(16) glm::vec4 emissiveColor;
    float metallic;
    float roughness;
    float transmission;
    uint32_t materialID;
    uint32_t activeAttributes;
    uint32_t vertexStrideWords;
    uint32_t uvOffsetWords;
    uint32_t normalOffsetWords;
};