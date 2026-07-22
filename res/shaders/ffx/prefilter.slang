[[vk::binding(0, 0)]] Texture2D<float4> g_InputRadiance; // Reprojected Radiance
[[vk::binding(1, 0)]] Texture2D<float>  g_InputVariance; // Reprojected Variance
[[vk::binding(2, 0)]] Texture2D<float4> g_InputNormal;
[[vk::binding(3, 0)]] Texture2D<float>  g_InputDepth;
[[vk::binding(4, 0)]] Texture2D<float>  g_InputRoughness;

[[vk::binding(5, 0)]] Texture2D<float4> g_InputAverageRadiance;

[[vk::binding(6, 0)]] RWTexture2D<float4> g_OutputRadiance;
[[vk::binding(7, 0)]] RWTexture2D<float>  g_OutputVariance;

[[vk::binding(8, 0)]] SamplerState g_LinearSampler;

struct FFXConstants {
    float4x4 invViewProjection;
    float4x4 projection;
    float4x4 invProjection;
    float4x4 prevViewProjection;
    uint2    renderSize;
    float    temporalStabilityFactor;
    int      maxSamples;
};
[[vk::binding(9, 0)]] ConstantBuffer<FFXConstants> cb;

#define min16float float
#define min16float2 float2
#define min16float3 float3
#define min16float4 float4

void FFX_DNSR_Reflections_LoadNeighborhood(int2 pixel, out float3 radiance, out float variance, out float3 normal, out float depth, uint2 screen_size) {
    radiance = g_InputRadiance.Load(int3(pixel, 0)).rgb;
    variance = g_InputVariance.Load(int3(pixel, 0)).r;
    normal = g_InputNormal.Load(int3(pixel, 0)).xyz * 2.0 - 1.0;

    float rawDepth = g_InputDepth.Load(int3(pixel, 0)).r;
    if (rawDepth >= 1.0) depth = 1e5;
    float4 clipPos = float4(0.0, 0.0, rawDepth, 1.0);
    float4 viewPos = mul(cb.invProjection, clipPos);
    depth = abs(viewPos.z / viewPos.w);
}

float FFX_DNSR_Reflections_LoadRoughness(int2 pixel) {
    return g_InputRoughness.Load(int3(pixel, 0)).r;
}

bool FFX_DNSR_Reflections_IsGlossyReflection(float roughness) {
    return roughness <= 1.0;
}

bool FFX_DNSR_Reflections_IsMirrorReflection(float roughness) {
    return roughness < 0.0001;
}

float3 FFX_DNSR_Reflections_SampleAverageRadiance(float2 uv) {
    return g_InputAverageRadiance.SampleLevel(g_LinearSampler, uv, 0).rgb;
}

void FFX_DNSR_Reflections_StorePrefilteredReflections(int2 pixel, float3 radiance, float variance) {
    g_OutputRadiance[pixel] = float4(radiance, 1.0);
    g_OutputVariance[pixel] = variance;
}

#include "Include/ffx-reflection-dnsr/ffx_denoiser_reflections_prefilter.h"

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID, uint3 groupThreadId : SV_GroupThreadID) {
    if (dispatchThreadId.x >= cb.renderSize.x || dispatchThreadId.y >= cb.renderSize.y) {
        return;
    }

    FFX_DNSR_Reflections_Prefilter(
        dispatchThreadId.xy,
        groupThreadId.xy,
        cb.renderSize
    );
}
