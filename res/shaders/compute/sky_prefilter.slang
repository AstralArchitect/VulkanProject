import clouds;
import cloud_common;

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
[vk::binding(0, 1)] RWTexture2DArray<float4> outputCubemap;
[vk::binding(1, 1)] SamplerCube inputSkyCubemapMip0;

struct PushConstants {
    uint faceSize;
    uint mipLevel;
    uint maxMipLevel;
    float roughness;
};

[vk::push_constant] PushConstants push;

float3 getCubemapDirection(uint2 id, uint face, uint size) {
    float2 st = (float2(id) + 0.5) / float(size);
    float2 uv = st * 2.0 - 1.0;
    uv.y = -uv.y;

    float3 dir = float3(0.0);
    switch (face) {
        case 0: dir = float3( 1.0,  uv.y, -uv.x); break; // +X
        case 1: dir = float3(-1.0,  uv.y,  uv.x); break; // -X
        case 2: dir = float3( uv.x,  1.0, -uv.y); break; // +Y
        case 3: dir = float3( uv.x, -1.0,  uv.y); break; // -Y
        case 4: dir = float3( uv.x,  uv.y,  1.0); break; // +Z
        case 5: dir = float3(-uv.x,  uv.y, -1.0); break; // -Z
    }
    return normalize(dir);
}

float2 Hammersley(uint i, uint N) {
    uint bits = (i << 16u) | (i >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return float2(float(i) / float(N), float(bits) * 2.3283064365386963e-10);
}

float3 ImportanceSampleGGX(float2 Xi, float3 N, float roughness) {
    float a = roughness * roughness;
    float phi = 2.0 * 3.14159265359 * Xi.x;
    float cosTheta = sqrt((1.0 - Xi.y) / (1.0 + (a * a - 1.0) * Xi.y));
    float sinTheta = sqrt(1.0 - cosTheta * cosTheta);

    float3 H;
    H.x = cos(phi) * sinTheta;
    H.y = sin(phi) * sinTheta;
    H.z = cosTheta;

    float3 up = abs(N.z) < 0.999 ? float3(0.0, 0.0, 1.0) : float3(1.0, 0.0, 0.0);
    float3 tangent = normalize(cross(up, N));
    float3 bitangent = cross(N, tangent);

    return normalize(tangent * H.x + bitangent * H.y + N * H.z);
}

// Calcul d'étoiles sous forme de points ronds ultra-nets
float3 calculateStars(float3 rayDir, float nightFactor, float time) {
    if (nightFactor <= 0.0) return float3(0.0);

    // Échelle de la sphère céleste (plus la valeur est grande, plus les points sont petits)
    float3 p = normalize(rayDir) * 180.0; 
    float3 cell = floor(p);
    float3 posInCell = frac(p);

    float starIntensity = 0.0;
    float3 starColor = float3(1.0);

    // Parcourir la cellule et ses voisines pour éviter les coupures aux bords
    [unroll]
    for (int z = 0; z <= 1; ++z) {
        [unroll]
        for (int y = 0; y <= 1; ++y) {
            [unroll]
            for (int x = 0; x <= 1; ++x) {
                float3 offsetCell = float3(x, y, z);
                float3 node = cell + offsetCell;

                // Sélectionner aléatoirement seulement 3% des cellules
                float starChance = hash31(node);
                if (starChance > 0.97) {
                    // Position aléatoire du centre de l'étoile dans la cellule
                    float3 starPos = offsetCell + float3(
                        hash31(node + float3(1.0, 0.0, 0.0)),
                        hash31(node + float3(0.0, 1.0, 0.0)),
                        hash31(node + float3(0.0, 0.0, 1.0))
                    );

                    // Distance au centre de l'étoile
                    float dist = length(posInCell - starPos);

                    // Atténuation radiale exponentielle très raide -> forme un point minuscule
                    float pointDot = exp(-dist * dist * 350.0);

                    // Scintillement individuel au fil du temps
                    float twinkle = sin(time * 3.5 + starChance * 62.8) * 0.35 + 0.65;
                    
                    // Variation de teinte (bleutée, blanche ou chaude)
                    float3 color = lerp(float3(0.7, 0.85, 1.0), float3(1.0, 0.9, 0.7), hash31(node + 5.0));

                    starIntensity += pointDot * twinkle;
                    starColor = color;
                }
            }
        }
    }

    return starColor * starIntensity * nightFactor * 3.0; // Intensité HDR des étoiles
}

// Fonction de calcul de la couleur du ciel atmosphérique
float3 calculateProceduralSky(float3 rayDir, float3 sunDir, float3 sunColor, int mipLevel) {
    float3 v = normalize(rayDir);
    float3 s = normalize(sunDir);

    float cosTheta = dot(v, s);
    float height = max(v.y, 0.0); // Hauteur dans le ciel (0 a l'horizon, 1 au zenith)

    float sunElevation = s.y;

    float nightFactor = saturate(-sunElevation * 4.0);

    // Day Sky
    float3 skyColorZenith = lerp(float3(0.05, 0.08, 0.25), float3(0.18, 0.42, 0.85), max(sunElevation, 0.0));

    float3 skyColorHorizon = lerp(float3(0.9, 0.35, 0.1), float3(0.7, 0.85, 0.95), max(sunElevation, 0.0));

    float3 skyBase = lerp(skyColorHorizon, skyColorZenith, pow(height, 0.5));

    float g = 0.85; // Facteur d'asymetrie de Mie
    float miePhase = (1.0 - g * g) / (4.0 * 3.14159 * pow(1.0 + g * g - 2.0 * g * cosTheta, 1.5));
    float3 mieColor = float3(1.0, 0.9, 0.7) * miePhase * 0.15;

    float sunAngularSize = 0.9995; // Controle la taille du disque solaire
    float sunDisk = smoothstep(sunAngularSize, sunAngularSize + 0.0002, cosTheta);
    float3 sunColor = sunColor * sunDisk; // Lumiere intense (HDR)

    float horizonFade = smoothstep(-0.2, 0.0, v.y);

    float3 daySky = (skyBase + mieColor + sunColor) * horizonFade;

    // Night sky
    float3 nightSkyColor = float3(0.001, 0.002, 0.006);
    float3 stars = calculateStars(v, nightFactor, camera.time);
    float3 moonDir = -s;
    float cosMoon = dot(v, normalize(moonDir));
    float moonDisk = smoothstep(0.9993, 0.9996, cosMoon) * nightFactor;
    float3 moonColor = float3(2.0, 2.2, 2.5) * moonDisk;
    float3 nightSky = (nightSkyColor + stars + moonColor) * horizonFade;

    float3 skyClear = lerp(daySky, nightSky, nightFactor);

    float4 cloud = calculateClouds(v, s, camera.sunColor.rgb, camera.time, nightFactor, mipLevel);

    return lerp(skyClear, cloud.rgb, cloud.a);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID) {
    uint face = dispatchThreadID.z;
    uint2 texel = dispatchThreadID.xy;

    if (texel.x >= push.faceSize || texel.y >= push.faceSize) return;

    float3 N = getCubemapDirection(texel, face, push.faceSize);

    if (push.mipLevel == 0 || push.roughness < 0.01) {
        float3 skyColor = calculateProceduralSky(N, camera.sunDir.xyz, camera.sunColor.rgb, push.mipLevel);
        outputCubemap[int3(texel, face)] = float4(skyColor, 1.0);
        return;
    }

    float3 R = N;
    float3 V = R;
    const uint SAMPLE_COUNT = 64u;
    float totalWeight = 0.0;
    float3 prefilteredColor = float3(0.0);

    for (uint i = 0u; i < SAMPLE_COUNT; ++i) {
        float2 Xi = Hammersley(i, SAMPLE_COUNT);
        float3 H = ImportanceSampleGGX(Xi, N, push.roughness);
        float3 L = normalize(2.0 * dot(V, H) * H - V);

        float NdotL = max(dot(N, L), 0.0);
        if (NdotL > 0.0) {
            float3 skySample = inputSkyCubemapMip0.SampleLevel(L, 0.0).rgb;
            
            prefilteredColor += skySample * NdotL;
            totalWeight += NdotL;
        }
    }

    prefilteredColor = prefilteredColor / max(totalWeight, 0.001);
    outputCubemap[int3(texel, face)] = float4(prefilteredColor, 1.0);
}