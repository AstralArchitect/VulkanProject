#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/hash.hpp>

struct PhysicsPose {
    glm::vec3 position;
    glm::quat orientation;

    glm::mat4 to_matrix() const {
        return glm::translate(glm::mat4(1.0f), position) * glm::mat4_cast(orientation);
    }
};

class PhysicsWorld {
public:
    virtual ~PhysicsWorld() = default;

    // Global lifecycle — call once at app start/shutdown, before/after any instance.
    static void global_init();
    static void global_shutdown();
    static std::unique_ptr<PhysicsWorld> create(); // Returns a JoltPhysicsWorld

    // Body management
    virtual JPH::BodyID create_body(const JPH::BodyCreationSettings& settings) = 0;
    virtual void        destroy_body(JPH::BodyID body_id) = 0;
    virtual void        set_motion_type(JPH::BodyID body_id, JPH::EMotionType type) = 0;
    virtual void        set_object_layer(JPH::BodyID body_id, uint16_t layer) = 0;
    virtual void        activate_body(JPH::BodyID body_id) = 0;

    // Kinematic/dynamic sync
    virtual void        move_kinematic(JPH::BodyID body_id, const PhysicsPose& pose) = 0;
    virtual PhysicsPose get_body_pose(JPH::BodyID body_id) const = 0;
    virtual glm::vec3   get_linear_velocity(JPH::BodyID body_id) const = 0;
    virtual void        set_linear_velocity(JPH::BodyID body_id, const glm::vec3& velocity) = 0;

    // Constraints (angles in radians)
    virtual void create_ball_socket_constraint(JPH::BodyID p1, JPH::BodyID p2,
                                               float swing_rad, float twist_rad) = 0;
    virtual void create_hinge_constraint(JPH::BodyID p1, JPH::BodyID p2,
                                         const glm::vec3& axis,
                                         float min_angle_rad, float max_angle_rad) = 0;

    // Simulation
    virtual void step(float delta_seconds) = 0;

    // Queries
    virtual bool raycast(const glm::vec3& origin, const glm::vec3& direction, float max_distance,
                         float& out_distance, glm::vec3& out_normal,
                         JPH::BodyID& out_body_id) const = 0;
};