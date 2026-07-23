[vk::binding(0, 0)] Texture2D<float4> inDirectColor;
[vk::binding(1, 0)] Texture2D<float4> inReflection;
[vk::binding(2, 0)] Texture2D<float4> inNormal;
[vk::binding(3, 0)] Texture2D<float> inDepth;

[[vk::binding(4, 0)]] RWTexture2D<float4> outFinalImage;
[vk::binding(5, 0)] Texture2D<float4> inRoughness;

float luminance(float3 color)
{
    return dot(color, float3(0.2126, 0.7152, 0.0722));
}

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 threadId : SV_DispatchThreadID)
{
    int2 texCoords = int2(threadId.xy);
    
    uint width, height;
    outFinalImage.GetDimensions(width, height);
    if (int(texCoords.x) >= int(width) || int(texCoords.y) >= int(height)) return;

    float4 directColor = inDirectColor.Load(int3(texCoords, 0));
    float4 reflection = inReflection.Load(int3(texCoords, 0));
    float roughness = inRoughness.Load(int3(texCoords, 0)).r;
    float centerDepth = inDepth.Load(int3(texCoords, 0)).r;
    float3 centerNormal = inNormal.Load(int3(texCoords, 0)).rgb * 2.0 - 1.0;

    float3 finalReflection = reflection.rgb;

    // Apply Gaussian blur to reflection if surface is rough (> 0.02) and not background (depth < 1.0)
    if (roughness > 0.02 && centerDepth < 1.0)
    {
        // Compute blur radius and Gaussian sigma based on material roughness
        float maxRadius = 16.0; // Maximum blur radius in pixels for roughness = 1.0
        float radius = roughness * maxRadius;
        float sigma = max(0.5, radius * 0.5);
        float twoSigmaSq = 2.0 * sigma * sigma;

        int intRadius = clamp(int(ceil(radius)), 1, 16);

        // Adaptive step size to maintain high performance while keeping blur smooth
        int step = max(1, intRadius / 6);

        float3 blurredReflection = float3(0.0);
        float totalWeight = 0.0;

        for (int dy = -intRadius; dy <= intRadius; dy += step)
        {
            int y = clamp(texCoords.y + dy, 0, int(height) - 1);

            for (int dx = -intRadius; dx <= intRadius; dx += step)
            {
                int x = clamp(texCoords.x + dx, 0, int(width) - 1);

                // Gaussian spatial weight based on exact pixel distance
                float distSq = float(dx * dx + dy * dy);
                float spatialWeight = exp(-distSq / twoSigmaSq);

                float3 sampleRefl = inReflection.Load(int3(x, y, 0)).rgb;
                float3 sampleNorm = inNormal.Load(int3(x, y, 0)).rgb * 2.0 - 1.0;
                float sampleDepth = inDepth.Load(int3(x, y, 0)).r;

                // Bilateral weights: geometry preservation (normal & depth)
                // Gentle normal weight (power 6.0) allows smooth blur on curved surfaces like spheres while preserving sharp object edges
                float normalWeight = pow(max(0.0, dot(centerNormal, sampleNorm)), 6.0);

                // Depth weight prevents bleeding across distinct geometry depth edges
                float depthDiff = abs(centerDepth - sampleDepth);
                float depthWeight = exp(-depthDiff * 80.0);

                float w = spatialWeight * normalWeight * depthWeight;

                blurredReflection += sampleRefl * w;
                totalWeight += w;
            }
        }

        if (totalWeight > 0.0)
        {
            blurredReflection /= totalWeight;
        }

        // Blend factor scales smoothly with roughness
        float blendFactor = saturate(roughness / 0.8);
        finalReflection = lerp(reflection.rgb, blurredReflection, blendFactor);
    }

    // Composition finale
    float3 color = directColor.rgb + finalReflection;

    // Tonemapping de base (Reinhard) + Gamma correction
    color = color / (color + float3(1.0));
    color = pow(color, float3(1.0 / 2.2));

    outFinalImage[texCoords] = float4(color, 1.0);
}