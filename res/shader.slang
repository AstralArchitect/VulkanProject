struct VSInput {
    float3 inPos;
    float3 inNormal;
    float2 inTexCoord;
};

struct VSOutput {
    float4 pos : SV_Position;
    float3 fragNormal;
    float2 fragTexCoord;
};

struct CameraUBO {
    float4x4 view;
    float4x4 proj;
};
[vk::binding(0, 0)] ConstantBuffer<CameraUBO> camera;

struct ModelPushConstants {
    float4x4 model;
    uint albedoTextureIndex;
    float4 baseColor;
    uint activeAttributes;
};

[vk::push_constant] ModelPushConstants push;

[shader("vertex")]
VSOutput vertMain(VSInput input) {
    VSOutput output;
    
    float4 worldPos = mul(push.model, float4(input.inPos, 1.0));
    float4 viewPos  = mul(camera.view, worldPos);
    output.pos      = mul(camera.proj, viewPos);
    
    output.fragNormal = input.inNormal;
    output.fragTexCoord = input.inTexCoord;
    
    return output;
}

[vk::binding(1, 1)] Sampler2D globalTextures[100];

[shader("fragment")]
float4 fragMain(VSOutput vertIn) : SV_TARGET {
    if ((push.activeAttributes & 1u) != 0u) {
        return globalTextures[push.albedoTextureIndex].Sample(vertIn.fragTexCoord);
    }

    return float4(push.baseColor);
}