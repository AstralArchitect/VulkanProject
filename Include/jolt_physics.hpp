#include "physics_world.hpp"

#include <Jolt/Jolt.h>
#include <Jolt/RegisterTypes.h>
#include <Jolt/Core/Factory.h>
#include <Jolt/Core/JobSystemThreadPool.h>
#include <Jolt/Core/TempAllocator.h>
#include <Jolt/Physics/PhysicsSystem.h>
#include <Jolt/Physics/Collision/RayCast.h>
#include <Jolt/Physics/Collision/CastResult.h>
#include <Jolt/Physics/Body/Body.h>
#include <Jolt/Physics/Body/BodyLock.h>
#include <Jolt/Physics/Constraints/SwingTwistConstraint.h>
#include <Jolt/Physics/Constraints/HingeConstraint.h>

// JoltPhysicsWorld owns its JPH::PhysicsSystem (does not take a pointer to one).
// Call PhysicsWorld::create() rather than constructing this directly.

namespace Layers {
    static constexpr JPH::ObjectLayer NON_MOVING = 0;
    static constexpr JPH::ObjectLayer MOVING = 1;
    static constexpr JPH::ObjectLayer NUM_LAYERS = 2;
};

namespace BroadPhaseLayers {
    static constexpr JPH::BroadPhaseLayer NON_MOVING(0);
    static constexpr JPH::BroadPhaseLayer MOVING(1);
    static constexpr uint32_t NUM_LAYERS(2);
};

class BPLayerInterfaceImpl final : public JPH::BroadPhaseLayerInterface {
public:
    BPLayerInterfaceImpl() {
        mObjectToBroadPhase[Layers::NON_MOVING] = BroadPhaseLayers::NON_MOVING;
        mObjectToBroadPhase[Layers::MOVING]     = BroadPhaseLayers::MOVING;
    }
    uint32_t GetNumBroadPhaseLayers() const override {
        return BroadPhaseLayers::NUM_LAYERS;
    }
    JPH::BroadPhaseLayer GetBroadPhaseLayer(JPH::ObjectLayer inLayer) const override {
        JPH_ASSERT(inLayer < Layers::NUM_LAYERS);
        return mObjectToBroadPhase[inLayer];
    }
#if defined(JPH_EXTERNAL_PROFILE) || defined(JPH_PROFILE_ENABLED)
    const char *GetBroadPhaseLayerName(JPH::BroadPhaseLayer inLayer) const override {
        switch ((JPH::BroadPhaseLayer::Type)inLayer) {
            case (JPH::BroadPhaseLayer::Type)BroadPhaseLayers::NON_MOVING: return "NON_MOVING";
            case (JPH::BroadPhaseLayer::Type)BroadPhaseLayers::MOVING:     return "MOVING";
            default:                                                       JPH_ASSERT(false); return "INVALID";
        }
    }
#endif // JPH_EXTERNAL_PROFILE || JPH_PROFILE_ENABLED
private:
    JPH::BroadPhaseLayer mObjectToBroadPhase[Layers::NUM_LAYERS];
};

class ObjVsBPFilter final : public JPH::ObjectVsBroadPhaseLayerFilter {
public:
    bool ShouldCollide(JPH::ObjectLayer inLayer1, JPH::BroadPhaseLayer inLayer2) const override {
        switch (inLayer1) {
            case Layers::NON_MOVING:
                return inLayer2 == BroadPhaseLayers::MOVING;
            case Layers::MOVING:
                return true;
            default:
                JPH_ASSERT(false);
                return false;
        }
    }
};

class ObjVsObjFilter final : public JPH::ObjectLayerPairFilter {
public:
    bool ShouldCollide(JPH::ObjectLayer inObject1, JPH::ObjectLayer inObject2) const override {
        switch (inObject1) {
            case Layers::NON_MOVING:
                return inObject2 == Layers::MOVING; // Non moving only collides with moving
            case Layers::MOVING:
                return true; // Moving collides with everything
            default:
                JPH_ASSERT(false);
                return false;
        }
    }
};

class JoltPhysicsWorld final : public PhysicsWorld {
public:
    explicit JoltPhysicsWorld(
        uint32_t max_bodies             = 20480,
        uint32_t max_body_pairs         = 65536,
        uint32_t max_contact_constraints = 32768,
        uint32_t num_worker_threads     = 4)
    {
        temp_allocator_ = std::make_unique<JPH::TempAllocatorImpl>(32 * 1024 * 1024);
        job_system_     = std::make_unique<JPH::JobSystemThreadPool>(
            JPH::cMaxPhysicsJobs, JPH::cMaxPhysicsBarriers, num_worker_threads);

        system_.Init(max_bodies, 0, max_body_pairs, max_contact_constraints,
                     bp_layer_interface_, obj_vs_bp_filter_, obj_vs_obj_filter_);
        body_interface_ = &system_.GetBodyInterface();
    }

    JPH::BodyID create_body(const JPH::BodyCreationSettings& settings) override {
        JPH::Body* body = body_interface_->CreateBody(settings);
        if (!body) return JPH::BodyID(); // allocation failed
        body_interface_->AddBody(body->GetID(), JPH::EActivation::Activate);
        return body->GetID();
    }

    void destroy_body(JPH::BodyID id) override {
        body_interface_->RemoveBody(id);
        body_interface_->DestroyBody(id);
    }

    void set_motion_type(JPH::BodyID id, JPH::EMotionType type) override {
        body_interface_->SetMotionType(id, type, JPH::EActivation::Activate);
    }

    void set_object_layer(JPH::BodyID id, uint16_t layer) override {
        body_interface_->SetObjectLayer(id, layer);
    }

    void activate_body(JPH::BodyID id) override {
        body_interface_->ActivateBody(id);
    }

    void move_kinematic(JPH::BodyID id, const PhysicsPose& pose) override {
        // JPH::RVec3 is used for world-space positions (double precision in large-world builds).
        body_interface_->SetPositionAndRotation(
            id,
            JPH::RVec3(pose.position.x, pose.position.y, pose.position.z),
            JPH::Quat(pose.orientation.x, pose.orientation.y,
                      pose.orientation.z, pose.orientation.w),
            JPH::EActivation::Activate);
    }

    PhysicsPose get_body_pose(JPH::BodyID id) const override {
        JPH::RVec3 pos = body_interface_->GetPosition(id);
        JPH::Quat  rot = body_interface_->GetRotation(id);
        return {
            glm::vec3(pos.GetX(), pos.GetY(), pos.GetZ()),
            glm::quat(rot.GetW(), rot.GetX(), rot.GetY(), rot.GetZ()),
        };
    }

    glm::vec3 get_linear_velocity(JPH::BodyID id) const override {
        JPH::Vec3 v = body_interface_->GetLinearVelocity(id);
        return glm::vec3(v.GetX(), v.GetY(), v.GetZ());
    }

    void set_linear_velocity(JPH::BodyID id, const glm::vec3& v) override {
        body_interface_->SetLinearVelocity(id, JPH::Vec3(v.x, v.y, v.z));
    }

    // Ball-socket joints use SwingTwistConstraint (not PointConstraint).
    // Body access requires BodyLockWrite for thread safety.
    void create_ball_socket_constraint(JPH::BodyID p1, JPH::BodyID p2,
                                       float swing_rad, float twist_rad) override {
        JPH::SwingTwistConstraintSettings s;
        s.mSpace               = JPH::EConstraintSpace::LocalToBodyCOM;
        s.mNormalHalfConeAngle = swing_rad;
        s.mPlaneHalfConeAngle  = swing_rad;
        s.mTwistMinAngle       = -twist_rad;
        s.mTwistMaxAngle       =  twist_rad;

        JPH::BodyLockWrite lock1(system_.GetBodyLockInterface(), p1);
        JPH::BodyLockWrite lock2(system_.GetBodyLockInterface(), p2);
        if (lock1.Succeeded() && lock2.Succeeded()) {
            auto* c = static_cast<JPH::SwingTwistConstraint*>(
                s.Create(lock1.GetBody(), lock2.GetBody()));
            system_.AddConstraint(c);
        }
    }

    void create_hinge_constraint(JPH::BodyID p1, JPH::BodyID p2,
                                 const glm::vec3& axis,
                                 float min_angle_rad, float max_angle_rad) override {
        JPH::HingeConstraintSettings s;
        s.mSpace      = JPH::EConstraintSpace::WorldSpace;
        s.mHingeAxis1 = s.mHingeAxis2 = JPH::Vec3(axis.x, axis.y, axis.z);
        s.mNormalAxis1 = s.mNormalAxis2 = JPH::Vec3(0, 1, 0);
        s.mLimitsMin  = min_angle_rad;
        s.mLimitsMax  = max_angle_rad;

        JPH::BodyLockWrite lock1(system_.GetBodyLockInterface(), p1);
        JPH::BodyLockWrite lock2(system_.GetBodyLockInterface(), p2);
        if (lock1.Succeeded() && lock2.Succeeded()) {
            auto* c = static_cast<JPH::HingeConstraint*>(
                s.Create(lock1.GetBody(), lock2.GetBody()));
            system_.AddConstraint(c);
        }
    }

    void step(float delta_seconds) override {
        system_.Update(delta_seconds, 2, temp_allocator_.get(), job_system_.get());
    }

    bool raycast(const glm::vec3& origin, const glm::vec3& direction, float max_distance,
                 float& out_distance, glm::vec3& out_normal, JPH::BodyID& out_body_id) const override {
        JPH::RRayCast ray(JPH::RVec3(origin.x, origin.y, origin.z),
                          JPH::Vec3(direction.x, direction.y, direction.z) * max_distance);
        JPH::RayCastResult result;
        if (system_.GetNarrowPhaseQuery().CastRay(ray, result)) {
            out_distance = result.mFraction * max_distance;
            out_body_id  = result.mBodyID;
            JPH::BodyLockRead lock(system_.GetBodyLockInterface(), out_body_id);
            if (lock.Succeeded()) {
                JPH::Vec3 normal = lock.GetBody().GetWorldSpaceSurfaceNormal(result.mSubShapeID2, ray.GetPointOnRay(result.mFraction));
                out_normal = glm::vec3(normal.GetX(), normal.GetY(), normal.GetZ());
            } else {
                out_normal = glm::vec3(0, 1, 0);
            }
            return true;
        }
        return false;
    }

    void add_force(JPH::BodyID id, const glm::vec3& f) override {
        body_interface_->AddForce(id, JPH::Vec3(f.x, f.y, f.z));
    }

    void add_impulse(JPH::BodyID id, const glm::vec3& imp) override {
        body_interface_->AddImpulse(id, JPH::Vec3(imp.x, imp.y, imp.z));
    }

private:
    // (Layer interface objects must outlive system_ — declare first.)
    BPLayerInterfaceImpl bp_layer_interface_;
    ObjVsBPFilter        obj_vs_bp_filter_;
    ObjVsObjFilter       obj_vs_obj_filter_;

    std::unique_ptr<JPH::TempAllocatorImpl>   temp_allocator_;
    std::unique_ptr<JPH::JobSystemThreadPool> job_system_;
    JPH::PhysicsSystem  system_;
    JPH::BodyInterface* body_interface_ = nullptr;
};