#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/hash.hpp>

enum GameMovment {
    FORWARD,
    BACKWARD,
    LEFT,
    RIGHT,
    JUMP
};

class GameEngine {
public:
    GameEngine();

    void movment(GameMovment);
private:
    bool wasGrounded = true;
    int currentAnimIndex = -1; // -1: bind pose / idle, 0: landing, 1: running, 2: jumping/air
    float animStartTime = 0.0f;

    float lastJumpTime = 0.f;
};