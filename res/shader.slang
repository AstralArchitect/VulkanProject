struct VSInput {
    [[vk::location(0)]] float3 inPos;
    [[vk::location(1)]] float3 inNormal;
    [[vk::location(2)]] float2 inTexCoord;
};

struct VSOutput {
    float4 pos : SV_Position;
    float3 fragPos;
    float3 fragNormal;
    float2 fragTexCoord;
};

struct CameraUBO {
    float4x4 view;
    float4x4 proj;
    float4x4 prevViewProj;
    float4 camPos;
    float4 lightsPos[2];
    float time;
};

struct InstanceData {
    float *vertexBuffer;
    uint *indexBuffer;
    float4 baseColor;
    float4 emissiveColor;
    float metallic;
    float roughness;
    float transmission;
    uint materialID;
    uint activeAttributes;
    uint vertexStrideWords;
    uint uvOffsetWords;
    uint normalOffsetWords;
};

[vk::binding(1, 1)] Sampler2D globalTextures[100];

[vk::binding(0, 0)] ConstantBuffer<CameraUBO> camera;
[vk::binding(1, 0)] RaytracingAccelerationStructure accelerationStructure;
[vk::binding(2, 0)] StructuredBuffer<InstanceData> instanceDataBuffer;

struct ModelPushConstants {
    float4x4 model;
    uint albedoTextureIndex;
    uint rmTextureIndex;
    float4 baseColor;
    float4 emissiveColor;
    float metallicFactor;
    float roughnessFactor;
    float transmissionFactor;
    uint activeAttributes;
};

[vk::push_constant] ModelPushConstants push;

[shader("vertex")]
VSOutput vertMain(VSInput input) {
    VSOutput output;
    
    float4 worldPos = mul(push.model, float4(input.inPos, 1.0));
    float4 viewPos  = mul(camera.view, worldPos);
    output.pos      = mul(camera.proj, viewPos);

    output.fragPos = mul(push.model, float4(input.inPos, 1.0)).xyz;

    output.fragNormal = mul((float3x3)push.model, input.inNormal);
    output.fragTexCoord = input.inTexCoord;
    
    return output;
}

static const float constant = 1.0f;
static const float linear = 0.09f;
static const float quadratic = 0.032f;
static const float PI = 3.14159265359;
static const float EPSILON = 0.001f;
static const float NUM_SAMPLES = 1;

float calculateAttenuation(float3 pos1, float3 pos2) {
    float distance = length(pos2 - pos1);
    float attenuation = 1.0 / max(distance * distance, 0.0001);
    return attenuation;
}

float3 fresnelSchlick(float cosTheta, float3 F0)
{
    return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
}

float DistributionGGX(float3 N, float3 H, float roughness)
{
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH = max(dot(N, H), 0.0);
    float NdotH2 = NdotH * NdotH;
    float num = a2;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = PI * denom * denom;
    return num / denom;
}

float GeometrySchlickGGX(float NdotV, float roughness)
{
    float r = (roughness + 1.0);
    float k = (r * r) / 8.0;
    float num = NdotV;
    float denom = NdotV * (1.0 - k) + k;
    return num / denom;
}

float GeometrySmith(float3 N, float3 V, float3 L, float roughness)
{
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    float ggx2 = GeometrySchlickGGX(NdotV, roughness);
    float ggx1 = GeometrySchlickGGX(NdotL, roughness);
    return ggx1 * ggx2;
}

float3 shadow_intensity(float3 P, float3 lightDir, float3 lightPos)
{
    float3 lightTransmissionColor = float3(1.0);
    float3 currentOrigin = P;
    float maxDist = distance(lightPos, P);
    float3 dir = normalize(lightDir);

    for (int depth = 0; depth < 8; ++depth)
    {
        RayDesc shadowRayDesc;
        shadowRayDesc.Origin = currentOrigin;
        shadowRayDesc.Direction = dir;
        shadowRayDesc.TMin = EPSILON;
        shadowRayDesc.TMax = maxDist;

        RayQuery<RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES> sq;
        let rayFlags = RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES;

        sq.TraceRayInline(accelerationStructure, rayFlags, 0xFF, shadowRayDesc);

        while (sq.Proceed()) {}

        if (sq.CommittedStatus() == COMMITTED_TRIANGLE_HIT)
        {
            uint instanceID = sq.CommittedRayInstanceCustomIndex() + sq.CommittedGeometryIndex();
            InstanceData inst = instanceDataBuffer[instanceID];

            lightTransmissionColor *= inst.transmission;

            if (all(lightTransmissionColor < 0.001))
            {
                lightTransmissionColor = float3(0.0);
                break;
            }

            float hitT = sq.CommittedRayT();
            currentOrigin = currentOrigin + dir * (hitT + EPSILON);
            maxDist -= (hitT + EPSILON);

            if (maxDist <= EPSILON)
            {
                break;
            }
        }
        else
        {
            break;
        }
    }

    return 1.0 - lightTransmissionColor;
}

float2 intersection_uv(uint instanceID, uint primIndex, float2 barycentric) {
    InstanceData inst = instanceDataBuffer[instanceID];

    uint i0 = inst.indexBuffer[primIndex * 3 + 0];
    uint i1 = inst.indexBuffer[primIndex * 3 + 1];
    uint i2 = inst.indexBuffer[primIndex * 3 + 2];

    uint base0 = i0 * inst.vertexStrideWords + inst.uvOffsetWords;
    uint base1 = i1 * inst.vertexStrideWords + inst.uvOffsetWords;
    uint base2 = i2 * inst.vertexStrideWords + inst.uvOffsetWords;

    float2 uv0 = float2(inst.vertexBuffer[base0], inst.vertexBuffer[base0 + 1]);
    float2 uv1 = float2(inst.vertexBuffer[base1], inst.vertexBuffer[base1 + 1]);
    float2 uv2 = float2(inst.vertexBuffer[base2], inst.vertexBuffer[base2 + 1]);

    float3 bary = float3(1.0 - barycentric.x - barycentric.y, barycentric.x, barycentric.y);
    return uv0 * bary.x + uv1 * bary.y + uv2 * bary.z;
}

float3 intersection_normal(uint instanceID, uint primIndex, float2 barycentric) {
    InstanceData inst = instanceDataBuffer[instanceID];

    // Les indices des 3 sommets du triangle
    uint i0 = inst.indexBuffer[primIndex * 3 + 0];
    uint i1 = inst.indexBuffer[primIndex * 3 + 1];
    uint i2 = inst.indexBuffer[primIndex * 3 + 2];

    // Calcule des positions mémoire de base pour la normale de chaque sommet
    uint base0 = i0 * inst.vertexStrideWords + inst.normalOffsetWords;
    uint base1 = i1 * inst.vertexStrideWords + inst.normalOffsetWords;
    uint base2 = i2 * inst.vertexStrideWords + inst.normalOffsetWords;

    // Lecture des composantes X, Y, Z pour chaque normale
    float3 n0 = float3(inst.vertexBuffer[base0], inst.vertexBuffer[base0 + 1], inst.vertexBuffer[base0 + 2]);
    float3 n1 = float3(inst.vertexBuffer[base1], inst.vertexBuffer[base1 + 1], inst.vertexBuffer[base1 + 2]);
    float3 n2 = float3(inst.vertexBuffer[base2], inst.vertexBuffer[base2 + 1], inst.vertexBuffer[base2 + 2]);

    // Interpolation à l'aide des coordonnées barycentriques
    float3 bary = float3(1.0 - barycentric.x - barycentric.y, barycentric.x, barycentric.y);
    return normalize(n0 * bary.x + n1 * bary.y + n2 * bary.z);
}

uint pcg_hash(uint seed) {
    uint state = seed * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

float next_rand(inout uint seed) {
    seed = pcg_hash(seed);
    return float(seed) / 4294967296.0;
}

uint init_rand(uint x, uint y, float time) {
    uint frameCounter = uint(time * 60.0);
    uint seed = x + y * 1973u + frameCounter * 9277u;
    seed = (seed ^ 61) ^ (seed >> 16);
    seed *= 9;
    seed = seed ^ (seed >> 4);
    seed *= 0x27d4eb2d;
    seed = seed ^ (seed >> 15);
    return seed;
}

float3 ImportanceSampleGGX(float2 Xi, float roughness) {
    float alpha = roughness * roughness;
    float alpha2 = alpha * alpha;

    float phi = 2.0 * PI * Xi.x;
    float cosTheta = sqrt(max((1.0 - Xi.y) / (1.0 + (alpha2 - 1.0) * Xi.y), 0.0));
    float sinTheta = sqrt(max(1.0 - cosTheta * cosTheta, 0.0));

    float3 H;
    H.x = sinTheta * cos(phi);
    H.y = sinTheta * sin(phi);
    H.z = cosTheta;

    return H;
}

void getTangentSpace(float3 N, out float3 T, out float3 B) {
    float3 up = abs(N.z) < 0.999 ? float3(0.0, 0.0, 1.0) : float3(1.0, 0.0, 0.0);
    T = normalize(cross(up, N));
    B = cross(N, T);
}

float3 orient_to_world(float3 H_local, float3 N, float3 T, float3 B) {
    return H_local.x * T + H_local.y * B + H_local.z * N;
}

float3 pbr(float3 P, float3 N, float3 albedoFactor, float metallicFactor, float roughnessFactor, float transmission, float3 emission) {
    float3 V = normalize(camera.camPos.xyz - P);

    float3 Lo = float3(0.0);
    for (int i = 0; i < 1; ++i)
    {
        float3 L = normalize(camera.lightsPos[i].xyz - P);
        float3 H = normalize(V + L);

        float3 shadowColor = shadow_intensity(P, L, camera.lightsPos[i].xyz);
        if (all(shadowColor >= 0.999)) continue;

        float attenuation = calculateAttenuation(P, camera.lightsPos[i].xyz);
        float3 lightColor = float3(23.47, 21.31, 20.79);
        float3 radiance = lightColor * attenuation * (1.0 - shadowColor);

        float3 F0 = float3(0.04);
        F0 = lerp(F0, albedoFactor, metallicFactor);
        float3 F = fresnelSchlick(max(dot(H, V), 0.0), F0);

        float NDF = DistributionGGX(N, H, roughnessFactor);
        float G = GeometrySmith(N, V, L, roughnessFactor);

        float3 numerator = NDF * G * F;
        float denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0);
        float3 specular = numerator / max(denominator, 0.001);

        float3 kS = F;
        float3 kD = float3(1.0) - kS;
        kD *= 1.0 - metallicFactor;

        float NdotL = max(dot(N, L), 0.0);

        float backNdotL = max(dot(-N, L), 0.0);
        float3 translucency = backNdotL * transmission * albedoFactor;

        Lo += ((kD * albedoFactor / PI) * NdotL + specular * NdotL + translucency) * radiance;
    }
    float3 ambient = float3(0.03) * albedoFactor;

    return ambient + Lo + emission;
}

float4 apply_reflection(float3 P, float3 N, float3 albedo, float metallicFactor, float roughnessFactor, inout uint seed, out float outRayLength) {
    float3 V = normalize(camera.camPos.xyz - P);
    
    float3 accumulatedColor = float3(0.0);
    float accumulatedRayLength = 0.0;
    float totalWeight = 0.0;

    float3 Tangent;
    float3 Bitangent;
    getTangentSpace(N, Tangent, Bitangent);

    const int num_samples = (roughnessFactor < .1f) ? 1 : NUM_SAMPLES;

    for (int i = 0; i < num_samples; ++i) {
        float2 Xi = float2(next_rand(seed), next_rand(seed));
        float3 H_local = ImportanceSampleGGX(Xi, max(roughnessFactor, 0.01)); 
        float3 H = orient_to_world(H_local, N, Tangent, Bitangent);
        float3 L = normalize(reflect(-V, H));

        if (dot(N, L) <= 0.0) continue;

        RayDesc reflectionRayDesc;
        reflectionRayDesc.Origin = P;
        reflectionRayDesc.Direction = L;
        reflectionRayDesc.TMin = EPSILON;
        reflectionRayDesc.TMax = 1e4;

        RayQuery<RAY_FLAG_FORCE_OPAQUE> rq;
        let rayFlags = RAY_FLAG_FORCE_OPAQUE;

        rq.TraceRayInline(accelerationStructure, rayFlags, 0xFF, reflectionRayDesc);
        while (rq.Proceed()) {}

        float3 sampleColor = float3(0.0);

        if (rq.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
            uint instanceID = rq.CommittedRayInstanceCustomIndex() + rq.CommittedGeometryIndex();
            uint primIndex = rq.CommittedPrimitiveIndex();
            float2 bary = rq.CommittedTriangleBarycentrics();

            float hitT = rq.CommittedRayT();
            accumulatedRayLength += hitT;
            float3 hitPosition = reflectionRayDesc.Origin + hitT * reflectionRayDesc.Direction;

            InstanceData inst = instanceDataBuffer[instanceID];
            
            float4 intersectionColor;
            if ((inst.activeAttributes & 1u) != 0u) {
                float2 uv = intersection_uv(instanceID, primIndex, bary);
                intersectionColor = globalTextures[inst.materialID].SampleLevel(uv, 0);
            } else {
                intersectionColor = inst.baseColor;
            }

            if ((inst.activeAttributes & 4u) != 0u) {
                float3 normal = intersection_normal(instanceID, primIndex, bary);
                intersectionColor.rgb = pbr(hitPosition, normal, intersectionColor.xyz, inst.metallic, inst.roughness, inst.transmission, inst.emissiveColor.rgb);
            }

            sampleColor = intersectionColor.rgb;
        } else {
            sampleColor = float3(.0f);
            accumulatedRayLength += 10000.0;
        }

        // Fresnel calculation on the microfacet
        float3 F0 = float3(0.04);
        F0 = lerp(F0, albedo, metallicFactor);
        float3 F_env = fresnelSchlick(max(dot(H, V), 0.0), F0);

        accumulatedColor += sampleColor * F_env;
        totalWeight += 1.0;
    }

    if (totalWeight > 0.0) {
        outRayLength = accumulatedRayLength / totalWeight;
        return float4(accumulatedColor / totalWeight, 1.f);
    }

    outRayLength = 10000.0;
    return float4(0.f);
}

struct FragmentOutput
{
    float4 directColor : SV_Target0; // Target 0 (Ambiant + Lumière directe)
    float4 reflection : SV_Target1;  // Target 1 (Reflet bruité linéaire HDR)
    float4 normal : SV_Target2;      // Target 2 (Normales)
    float4 roughness : SV_Target3;   // On met float4 pour simplifier parce que j'ai la flemme de changer la fonction d'alloc des renderImages
    float4 motionVector : SV_Target4;
    float4 rayLength : SV_Target5;
};

[shader("fragment")]
FragmentOutput fragMain(VSOutput vertIn) : SV_TARGET {
    FragmentOutput output;

    float4 color;
    float3 P = vertIn.fragPos;
    uint seed = init_rand(uint(vertIn.pos.x), uint(vertIn.pos.y), camera.time);

    if ((push.activeAttributes & 4u) != 0u) {
        float3 albedoFactor;
        if ((push.activeAttributes & 1u) != 0u) {
            albedoFactor = globalTextures[push.albedoTextureIndex].Sample(vertIn.fragTexCoord).xyz;
        } else {
            albedoFactor = push.baseColor.xyz;
        }

        float metallicFactor;
        float roughnessFactor;

        if ((push.activeAttributes & 2u) != 0u) {
            float3 metallicRoughnessFactor = globalTextures[push.rmTextureIndex].Sample(vertIn.fragTexCoord).xyz;

            metallicFactor = metallicRoughnessFactor.z;
            roughnessFactor = metallicRoughnessFactor.y;
        } else {
            metallicFactor = push.metallicFactor;
            roughnessFactor = push.roughnessFactor;
        }

        float3 N = normalize(vertIn.fragNormal);

        color = float4(pbr(P, N, albedoFactor, metallicFactor, roughnessFactor, push.transmissionFactor, push.emissiveColor.rgb), 1.f);

        float rayLength = 0.0;
        float4 reflectionColor = apply_reflection(P, N, albedoFactor, metallicFactor, roughnessFactor, seed, rayLength);

        output.directColor = color;
        output.reflection = reflectionColor;
        output.normal = float4(N * 0.5 + 0.5, 1.0);
        output.roughness = float4(roughnessFactor, 0.0, 0.0, 0.0);
        output.rayLength = float4(rayLength, 0.0, 0.0, 0.0);
        
    } else if ((push.activeAttributes & 1u) != 0u) {
        color =  globalTextures[push.albedoTextureIndex].Sample(vertIn.fragTexCoord);

        output.directColor = color;
        output.reflection = float4(0.0);
        output.normal = float4(0.0);
        output.roughness = float4(0.0);
        output.rayLength = float4(10000.0, 0.0, 0.0, 0.0);
    }
    
    // Motion vector calculation
    float4 prevClipPos = mul(camera.prevViewProj, float4(P, 1.0));
    float2 prevNDC = prevClipPos.xy / prevClipPos.w;
    float4 currClipPos = mul(camera.proj, mul(camera.view, float4(P, 1.0)));
    float2 currNDC = currClipPos.xy / currClipPos.w;
    
    // Convert NDC to UV (Vulkan: invert Y via -0.5)
    float2 prevUV = prevNDC * 0.5 + 0.5;
    float2 currUV = currNDC * 0.5 + 0.5;
    float2 motionVector = currUV - prevUV;

    output.motionVector = float4(motionVector, 0.0, 0.0);

    return output;
}