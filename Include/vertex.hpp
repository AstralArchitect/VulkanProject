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
    uint32_t albedoTextureIndex;
    uint32_t rmTextureIndex;
    alignas(16) glm::vec4 baseColor;
    float metallicFactor;
    float roughnessFactor;
    uint32_t activeAttributes;
};

struct InstanceData {
    uint64_t vertexBufferAddress;
    uint64_t indexBufferAddress;
    glm::vec4 baseColor;
    uint32_t materialID;
    uint32_t activeAttributes;
    uint32_t vertexStrideWords;
    uint32_t uvOffsetWords;
};