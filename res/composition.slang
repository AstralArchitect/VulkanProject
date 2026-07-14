[vk::binding(0, 0)] Texture2D<float4> inDirectColor;
[vk::binding(1, 0)] Texture2D<float4> inReflection;
[vk::binding(2, 0)] Texture2D<float4> inNormal;
[vk::binding(3, 0)] Texture2D<float> inDepth;

[[vk::binding(4, 0)]] RWTexture2D<float4> outFinalImage;

static const int FILTER_RADIUS = 2;          // Rayon de recherche (2 = 5x5 pixels)
static const float SIGMA_SPATIAL = 2.0;      // Douceur spatiale (plus c'est élevé, plus ça floute loin)
static const float SIGMA_DEPTH = 0.02;       // Sensibilité à la profondeur (tolérance de distance)
static const float SIGMA_NORMAL = 128.0;     // Sensibilité aux normales (puissance du produit scalaire)

// Fonction de débruitage
float3 denoiseReflection(int2 texCoords, uint width, uint height) {
    float3 centerReflection = inReflection.Load(int3(texCoords, 0)).rgb;
    
    if (dot(centerReflection, centerReflection) == 0.0) {
        return centerReflection;
    }

    float3 centerNormal = inNormal.Load(int3(texCoords, 0)).xyz;
    float centerDepth = inDepth.Load(int3(texCoords, 0)).r;

    float3 sumColor = float3(0.0);
    float sumWeight = 0.0;

    for (int y = -FILTER_RADIUS; y <= FILTER_RADIUS; ++y) {
        for (int x = -FILTER_RADIUS; x <= FILTER_RADIUS; ++x) {
            
            int2 offset = int2(x, y);
            int2 neighborCoords = texCoords + offset;

            neighborCoords = clamp(neighborCoords, int2(0, 0), int2(width - 1, height - 1));

            float3 neighborReflection = inReflection.Load(int3(neighborCoords, 0)).rgb;
            float3 neighborNormal = inNormal.Load(int3(neighborCoords, 0)).xyz;
            float neighborDepth = inDepth.Load(int3(neighborCoords, 0)).r;

            float spatialDist2 = x * x + y * y;
            float spatialWeight = exp(-spatialDist2 / (2.0 * SIGMA_SPATIAL * SIGMA_SPATIAL));

            float depthDiff = abs(centerDepth - neighborDepth);
            float depthWeight = exp(-depthDiff / SIGMA_DEPTH);

            float normalDot = max(dot(centerNormal, neighborNormal), 0.0);
            float normalWeight = pow(normalDot, SIGMA_NORMAL);

            float weight = spatialWeight * depthWeight * normalWeight;

            sumColor += neighborReflection * weight;
            sumWeight += weight;
        }
    }

    return sumWeight > 0.0 ? (sumColor / sumWeight) : centerReflection;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 threadId : SV_DispatchThreadID)
{
    int2 texCoords = int2(threadId.xy);
    
    uint width, height;
    outFinalImage.GetDimensions(width, height);
    if (texCoords.x >= width || texCoords.y >= height) return;

    float4 directColor = inDirectColor.Load(int3(texCoords, 0));
    float4 reflection = inReflection.Load(int3(texCoords, 0));
    
    float3 filteredReflection = denoiseReflection(texCoords, width, height);

    float3 color = directColor.rgb + filteredReflection;

    // Tonemapping + Gamma correction
    color = color / (color + float3(1.0));
    color = pow(color, float3(1.0 / 2.2));

    outFinalImage[texCoords] = float4(color, 1.0);
}