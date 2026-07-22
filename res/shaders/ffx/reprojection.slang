[[vk::binding(0, 0)]] Texture2D<float4> g_InputRadiance;
[[vk::binding(1, 0)]] Texture2D<float>  g_InputDepth;
[[vk::binding(2, 0)]] Texture2D<float4> g_InputNormal;
[[vk::binding(3, 0)]] Texture2D<float>  g_InputRoughness;
[[vk::binding(4, 0)]] Texture2D<float2> g_InputMotionVectors;
[[vk::binding(5, 0)]] Texture2D<float>  g_InputRayLength;

[[vk::binding(6, 0)]] Texture2D<float4> g_HistoryRadiance;
[[vk::binding(7, 0)]] Texture2D<float>  g_HistoryDepth;
[[vk::binding(8, 0)]] Texture2D<float4> g_HistoryNormal;
[[vk::binding(9, 0)]] Texture2D<float>  g_HistoryRoughness;

[[vk::binding(10, 0)]] Texture2D<float> g_HistoryVariance;
[[vk::binding(11, 0)]] Texture2D<float> g_HistoryNumSamples;

[[vk::binding(12, 0)]] RWTexture2D<float4> g_OutputReprojectedRadiance;
[[vk::binding(13, 0)]] RWTexture2D<float4> g_OutputAverageRadiance;
[[vk::binding(14, 0)]] RWTexture2D<float>  g_OutputVariance;
[[vk::binding(15, 0)]] RWTexture2D<float>  g_OutputNumSamples;

[[vk::binding(16, 0)]] SamplerState g_LinearSampler;

struct FFXConstants {
    float4x4 invViewProjection;
    float4x4 projection;
    float4x4 invProjection;
    float4x4 prevViewProjection;
    uint2    renderSize;
    float    temporalStabilityFactor;
    int      maxSamples;
};
[[vk::binding(17, 0)]] ConstantBuffer<FFXConstants> cb;

#define min16float float
#define min16float2 float2
#define min16float3 float3
#define min16float4 float4

float3 FFX_DNSR_Reflections_LoadRadiance(int2 pixel) {
    return g_InputRadiance.Load(int3(pixel, 0)).rgb;
}
float FFX_DNSR_Reflections_LoadDepth(int2 pixel) {
    return g_InputDepth.Load(int3(pixel, 0)).r;
}
float2 FFX_DNSR_Reflections_LoadMotionVector(int2 pixel) {
    return g_InputMotionVectors.Load(int3(pixel, 0)).xy;
}
float3 FFX_DNSR_Reflections_LoadWorldSpaceNormal(int2 pixel) {
    return g_InputNormal.Load(int3(pixel, 0)).xyz * 2.0 - 1.0;
}
float FFX_DNSR_Reflections_LoadRoughness(int2 pixel) {
    return g_InputRoughness.Load(int3(pixel, 0)).r;
}
float FFX_DNSR_Reflections_LoadRayLength(int2 pixel) {
    return g_InputRayLength.Load(int3(pixel, 0)).r;
}

float3 FFX_DNSR_Reflections_SampleRadianceHistory(float2 uv) {
    return g_HistoryRadiance.SampleLevel(g_LinearSampler, uv, 0).rgb;
}
float FFX_DNSR_Reflections_SampleDepthHistory(float2 uv) {
    return g_HistoryDepth.SampleLevel(g_LinearSampler, uv, 0).r;
}
float3 FFX_DNSR_Reflections_SampleWorldSpaceNormalHistory(float2 uv) {
    return g_HistoryNormal.SampleLevel(g_LinearSampler, uv, 0).xyz * 2.0 - 1.0;
}
float FFX_DNSR_Reflections_SampleRoughnessHistory(float2 uv) {
    return g_HistoryRoughness.SampleLevel(g_LinearSampler, uv, 0).r;
}

float FFX_DNSR_Reflections_SampleVarianceHistory(float2 uv) {
    return g_HistoryVariance.SampleLevel(g_LinearSampler, uv, 0).r;
}

float FFX_DNSR_Reflections_SampleNumSamplesHistory(float2 uv) {
    return g_HistoryNumSamples.SampleLevel(g_LinearSampler, uv, 0).r;
}

float3 FFX_DNSR_Reflections_LoadRadianceHistory(int2 pixel) {
    return g_HistoryRadiance.Load(int3(pixel, 0)).rgb;
}
float3 FFX_DNSR_Reflections_LoadWorldSpaceNormalHistory(int2 pixel) {
    return g_HistoryNormal.Load(int3(pixel, 0)).xyz * 2.0 - 1.0;
}
float FFX_DNSR_Reflections_LoadDepthHistory(int2 pixel) {
    return g_HistoryDepth.Load(int3(pixel, 0)).r;
}

float FFX_DNSR_Reflections_GetLinearDepth(float2 uv, float depth) {
    if (depth >= 1.0) return 1e5;
    float4 clipSpace = float4(0.0, 0.0, depth, 1.0);
    float4 viewSpace = mul(cb.invProjection, clipSpace);
    return abs(viewSpace.z / viewSpace.w);
}

float3 FFX_DNSR_Reflections_ScreenSpaceToViewSpace(float3 screenSpace) {
    float2 clipUV = screenSpace.xy * 2.0 - 1.0;
    float4 viewSpace = mul(cb.invProjection, float4(clipUV, screenSpace.z, 1.0));
    return viewSpace.xyz / viewSpace.w;
}

float3 FFX_DNSR_Reflections_ViewSpaceToWorldSpace(float4 viewSpace) {
    float4 worldSpace = mul(cb.invViewProjection, mul(cb.projection, viewSpace));
    return worldSpace.xyz / worldSpace.w;
}

float3 FFX_DNSR_Reflections_WorldSpaceToScreenSpacePrevious(float3 worldSpacePos) {
    float4 clipSpace = mul(cb.prevViewProjection, float4(worldSpacePos, 1.0));
    clipSpace.xyz /= clipSpace.w;
    float2 uv = clipSpace.xy * 0.5 + 0.5;
    return float3(uv, clipSpace.z);
}

bool FFX_DNSR_Reflections_IsGlossyReflection(float roughness) {
    return roughness <= 1.0;
}

void FFX_DNSR_Reflections_StoreRadianceReprojected(int2 pixel, float3 radiance) {
    g_OutputReprojectedRadiance[pixel] = float4(radiance, 1.0);
}
void FFX_DNSR_Reflections_StoreVariance(int2 pixel, float variance) {
    g_OutputVariance[pixel] = variance;
}
void FFX_DNSR_Reflections_StoreNumSamples(int2 pixel, float num_samples) {
    g_OutputNumSamples[pixel] = num_samples;
}
void FFX_DNSR_Reflections_StoreAverageRadiance(int2 pixel, float3 radiance) {
    g_OutputAverageRadiance[pixel] = float4(radiance, 1.0);
}

#include "Include/ffx-reflection-dnsr/ffx_denoiser_reflections_reproject.h"

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID, uint3 groupThreadId : SV_GroupThreadID) {
    if (dispatchThreadId.x >= cb.renderSize.x || dispatchThreadId.y >= cb.renderSize.y) {
        return;
    }

    FFX_DNSR_Reflections_Reproject(
        dispatchThreadId.xy, 
        groupThreadId.xy, 
        cb.renderSize, 
        cb.temporalStabilityFactor, 
        cb.maxSamples
    );
}
