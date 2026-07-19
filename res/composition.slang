[vk::binding(0, 0)] Texture2D<float4> inDirectColor;
[vk::binding(1, 0)] Texture2D<float4> inReflection;
[vk::binding(2, 0)] Texture2D<float4> inNormal;
[vk::binding(3, 0)] Texture2D<float> inDepth;

[[vk::binding(4, 0)]] RWTexture2D<float4> outFinalImage;

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

    // Composition finale
    float3 color = directColor.rgb + reflection.rgb;

    // Tonemapping de base (Reinhard) + Gamma correction
    color = color / (color + float3(1.0));
    color = pow(color, float3(1.0 / 2.2));

    outFinalImage[texCoords] = float4(color, 1.0);
}