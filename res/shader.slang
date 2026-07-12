struct VSInput {
    float3 inPos;
    float3 inColor;
    float2 inTexCoord;
};

struct VSOutput {
    float4 pos : SV_Position;
    float3 fragColor;
    float2 fragTexCoord;
};

struct CameraUBO {
    float4x4 view;
    float4x4 proj;
};
[vk::binding(0, 0)] ConstantBuffer<CameraUBO> camera;

struct ModelPushConstants {
    float4x4 model;
};
[vk::push_constant] ConstantBuffer<ModelPushConstants> push;

[shader("vertex")]
VSOutput vertMain(VSInput input) {
    VSOutput output;
    
    float4 worldPos = mul(push.model, float4(input.inPos, 1.0));
    float4 viewPos  = mul(camera.view, worldPos);
    output.pos      = mul(camera.proj, viewPos);
    
    output.fragColor = input.inColor;
    output.fragTexCoord = input.inTexCoord;
    
    return output;
}

// [vk::binding(1, 0)] Sampler2D myTexture;

[shader("fragment")]
float4 fragMain(VSOutput vertIn) : SV_TARGET {
    return float4(vertIn.fragColor, 1.0);
    
    // return myTexture.Sample(vertIn.fragTexCoord);
}