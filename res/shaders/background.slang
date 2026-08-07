struct VSOutput {
    float4 pos : SV_Position;
    float3 rayDir;
};

struct CameraUBO {
    float4x4 view;
    float4x4 proj;
    float4x4 prevViewProj;
    float4 camPos;
    float time;
    float4 sunDir;
    float4 sunColor;
};

[vk::binding(0, 0)] ConstantBuffer<CameraUBO> camera;
[vk::binding(3, 0)] SamplerCube prefilteredSkyMap;

[shader("vertex")]
VSOutput vertMain(uint vertexID : SV_VertexID) {
    VSOutput output;
    
    // Generate full screen triangle using vertexID (0, 1, 2)
    float2 uv = float2((vertexID << 1) & 2, vertexID & 2);
    float4 clipPos = float4(uv * 2.0f - 1.0f, 1.0f, 1.0f); // Z=1.0 for skybox background
    output.pos = clipPos;
    
    // Unproject clip space pos to view space direction
    float3 viewDir = float3(clipPos.x / camera.proj[0][0], clipPos.y / camera.proj[1][1], -1.0);
    
    // Transform view space direction to world space using inverse of view rotation (transpose)
    float3x3 viewRot = float3x3(
        camera.view[0].xyz,
        camera.view[1].xyz,
        camera.view[2].xyz
    );
    output.rayDir = mul(viewDir, viewRot);
    
    return output;
}

[shader("fragment")]
float4 fragMain(VSOutput vertIn) : SV_TARGET0 {
    float3 color = prefilteredSkyMap.SampleLevel(normalize(vertIn.rayDir), 0.0).rgb;
    return float4(color, 1.0);
}