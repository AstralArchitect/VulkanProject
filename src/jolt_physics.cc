#include "jolt_physics.hpp"

#include <iostream>
#include <cstdarg>

static void TraceImpl(const char *inFMT, ...) {
    // Format the message
    va_list list;
    va_start(list, inFMT);
    char buffer[1024];
    vsnprintf(buffer, sizeof(buffer), inFMT, list);
    va_end(list);

    // Print to the TTY
    std::cout << buffer << std::endl;
}

#ifdef JPH_ENABLE_ASSERTS
static bool AssertFailedImpl(const char *inExpression, const char *inMessage, const char *inFile, uint32_t inLine) {
    std::cout << inFile << ":" << inLine << ": (" << inExpression << ") " << (inMessage != nullptr ? inMessage : "") << std::endl;
    return true; // Breakpoint
}
#endif // JPH_ENABLE_ASSERTS

void PhysicsWorld::global_init() {
    JPH::RegisterDefaultAllocator();
    
    JPH::Trace = TraceImpl;
    JPH_IF_ENABLE_ASSERTS(JPH::AssertFailed = AssertFailedImpl;)

    JPH::Factory::sInstance = new JPH::Factory();
    
    JPH::RegisterTypes();
}

void PhysicsWorld::global_shutdown() {
    JPH::UnregisterTypes();
    
    delete JPH::Factory::sInstance;
    JPH::Factory::sInstance = nullptr;
}

std::unique_ptr<PhysicsWorld> PhysicsWorld::create() {
    return std::make_unique<JoltPhysicsWorld>();
}
