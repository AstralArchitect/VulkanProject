#include <iostream>
#include <cstdint>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

struct InstanceData {
    uint64_t vertexBufferAddress; // 0
    uint64_t indexBufferAddress; // 8
    glm::vec4 baseColor; // 16
    uint32_t materialID; // 32
    uint32_t activeAttributes; // 36
    uint32_t vertexStrideWords; // 40
    uint32_t uvOffsetWords; // 44
    uint32_t normalOffsetWords; // 48
    uint32_t pad0;
    uint32_t pad1;
    uint32_t pad2;
};

int main() {
    std::cout << "Size of InstanceData: " << sizeof(InstanceData) << std::endl;
    std::cout << "Offset of baseColor: " << offsetof(InstanceData, baseColor) << std::endl;
    std::cout << "Offset of normalOffsetWords: " << offsetof(InstanceData, normalOffsetWords) << std::endl;
    std::cout << "Offset of pad0: " << offsetof(InstanceData, pad0) << std::endl;
    return 0;
}
