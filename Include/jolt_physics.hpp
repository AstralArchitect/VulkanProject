#include "physics_world.hpp"

#include <Jolt/Jolt.h>
#include <Jolt/RegisterTypes.h>
#include <Jolt/Core/Factory.h>
#include <Jolt/Physics/PhysicsSystem.h>

// JoltPhysicsWorld owns its JPH::PhysicsSystem (does not take a pointer to one).
// Call PhysicsWorld::create() rather than constructing this directly.
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
        JPH::RVec3 pos;
        JPH::Quat  rot;
        body_interface_->GetPositionAndRotation(id, pos, rot);
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
        system_.Update(delta_seconds, 1, temp_allocator_.get(), job_system_.get());
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
            out_normal = lock.Succeeded()
                ? glm::vec3(lock.GetBody().GetWorldSpaceSurfaceNormal(
                      result.mSubShapeID2, ray.GetPointOnRay(result.mFraction)).GetX(),
                      lock.GetBody().GetWorldSpaceSurfaceNormal(
                      result.mSubShapeID2, ray.GetPointOnRay(result.mFraction)).GetY(),
                      lock.GetBody().GetWorldSpaceSurfaceNormal(
                      result.mSubShapeID2, ray.GetPointOnRay(result.mFraction)).GetZ())
                : glm::vec3(0, 1, 0);
            return true;
        }
        return false;
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