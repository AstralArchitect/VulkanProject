struct VSInput {
    float3 inPos;
    float3 inNormal;
    float2 inTexCoord;
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
    float4 camPos;
    float4 lightsPos[4];
};
[vk::binding(0, 0)] ConstantBuffer<CameraUBO> camera;

struct ModelPushConstants {
    float4x4 model;
    uint albedoTextureIndex;
    uint rmTextureIndex;
    float4 baseColor;
    float metallicFactor;
    float roughnessFactor;
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
static const float NUM_SAMPLES = 8;

[vk::binding(1, 1)] Sampler2D globalTextures[100];
[vk::binding(1, 0)] RaytracingAccelerationStructure accelerationStructure;

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

bool in_shadow(float3 P, float3 lightDir, float3 lightPos)
{
    // Build the shadow ray from the world position toward the light
    RayDesc shadowRayDesc;
    shadowRayDesc.Origin = P;
    shadowRayDesc.Direction = normalize(lightDir);
    shadowRayDesc.TMin = EPSILON;
    shadowRayDesc.TMax = distance(lightPos, P);

    // Initialize a ray query for shadows
    RayQuery<RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES |
             RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> sq;
    let rayFlags = RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES |
                   RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH;

    sq.TraceRayInline(accelerationStructure, rayFlags, 0xFF, shadowRayDesc);

    sq.Proceed();

    // If the shadow ray intersects an opaque triangle, we consider the pixel in shadow
    bool hit = (sq.CommittedStatus() == COMMITTED_TRIANGLE_HIT);

    return hit;
}

struct InstanceData {
    float* vertexBuffer;
    uint* indexBuffer;
    float4 baseColor;
    uint materialID;
    uint activeAttributes;
    uint vertexStrideWords;
    uint uvOffsetWords;
};

[vk::binding(2, 0)] StructuredBuffer<InstanceData> instanceDataBuffer;

float2 intersection_uv(uint instanceID, uint primIndex, float2 barycentric) {
    InstanceData inst = instanceDataBuffer[instanceID];

    // Récupération des 3 indices du triangle
    uint i0 = inst.indexBuffer[primIndex * 3 + 0];
    uint i1 = inst.indexBuffer[primIndex * 3 + 1];
    uint i2 = inst.indexBuffer[primIndex * 3 + 2];

    // Calcul de la position de départ pour les UV de chaque sommet
    uint base0 = i0 * inst.vertexStrideWords + inst.uvOffsetWords;
    uint base1 = i1 * inst.vertexStrideWords + inst.uvOffsetWords;
    uint base2 = i2 * inst.vertexStrideWords + inst.uvOffsetWords;

    // Lecture directe des 2 floats (U et V) dans le buffer
    float2 uv0 = float2(inst.vertexBuffer[base0], inst.vertexBuffer[base0 + 1]);
    float2 uv1 = float2(inst.vertexBuffer[base1], inst.vertexBuffer[base1 + 1]);
    float2 uv2 = float2(inst.vertexBuffer[base2], inst.vertexBuffer[base2 + 1]);

    // Interpolation barycentrique
    float3 bary = float3(1.0 - barycentric.x - barycentric.y, barycentric.x, barycentric.y);
    return uv0 * bary.x + uv1 * bary.y + uv2 * bary.z;
}

// --------------------------------------------------------------------------------
// Utilitaires Mathématiques pour Monte Carlo
// --------------------------------------------------------------------------------

uint pcg_hash(uint seed) {
    uint state = seed * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

float next_rand(inout uint seed) {
    seed = pcg_hash(seed);
    return float(seed) / 4294967296.0;
}

uint init_rand(float x, float y) {
    return uint(x) * 1973u + uint(y) * 9277u;
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


void apply_reflection(float3 P, float3 N, inout float4 baseColor, float3 albedo, float metallicFactor, float roughnessFactor, inout uint seed) {
    float3 V = normalize(camera.camPos.xyz - P);
    
    const int NUM_SAMPLES = 32;
    float3 accumulatedColor = float3(0.0);
    float totalWeight = 0.0;

    float3 Tangent;
    float3 Bitangent;
    getTangentSpace(N, Tangent, Bitangent);

    for (int i = 0; i < NUM_SAMPLES; ++i) {
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
            uint instanceID = rq.CommittedRayInstanceCustomIndex();
            uint primIndex = rq.CommittedPrimitiveIndex();
            float2 bary = rq.CommittedTriangleBarycentrics();

            InstanceData inst = instanceDataBuffer[instanceID];
            
            float4 intersectionColor;
            if ((inst.activeAttributes & 1u) != 0u) {
                float2 uv = intersection_uv(instanceID, primIndex, bary);
                intersectionColor = globalTextures[inst.materialID].SampleLevel(uv, 0);
            } else {
                intersectionColor = inst.baseColor;
            }

            sampleColor = intersectionColor.rgb;
        }

        // Fresnel calculation on the microfacet
        float3 F0 = float3(0.04);
        F0 = lerp(F0, albedo, metallicFactor);
        float3 F_env = fresnelSchlick(max(dot(H, V), 0.0), F0);

        accumulatedColor += sampleColor * F_env;
        totalWeight += 1.0;
    }

    if (totalWeight > 0.0) {
        baseColor.rgb += accumulatedColor / totalWeight;
    }
}

[shader("fragment")]
float4 fragMain(VSOutput vertIn) : SV_TARGET {
    float4 color;
    float3 P = vertIn.fragPos;
    uint seed = init_rand(vertIn.pos.x, vertIn.pos.y);

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

            metallicFactor = metallicRoughnessFactor.y;
            roughnessFactor = metallicRoughnessFactor.x;
        } else {
            metallicFactor = push.metallicFactor;
            roughnessFactor = push.roughnessFactor;
        }

        float3 N = normalize(vertIn.fragNormal);
        float3 V = normalize(camera.camPos.xyz - vertIn.fragPos);

        float3 Lo = float3(0.0);
        for (int i = 0; i < 4; ++i)
        {
            float3 L = normalize(camera.lightsPos[i].xyz - vertIn.fragPos);
            float3 H = normalize(V + L);

            // First, test if the object is in shadow or not. If not, we don't add this light's contribution
            bool inShadow = in_shadow(P, L, camera.lightsPos[i].xyz);

            if (inShadow) continue;

            float attenuation = calculateAttenuation(vertIn.fragPos, camera.lightsPos[i].xzy);
            float3 lightColor = float3(23.47, 21.31, 20.79);
            float3 radiance = lightColor * attenuation;

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
            Lo += (kD * albedoFactor / PI + specular) * radiance * NdotL;
        }
        float3 ambient = float3(0.03) * albedoFactor;
        color = float4(ambient + Lo, 1.f);

        if (roughnessFactor < 1.f) {
            apply_reflection(P, N, color, albedoFactor, metallicFactor, roughnessFactor, seed);
        }

        color = color / (color + float4(1.0));
        color = pow(color, float4(1.0 / 2.2));

    } else if ((push.activeAttributes & 1u) != 0u) {
        color =  globalTextures[push.albedoTextureIndex].Sample(vertIn.fragTexCoord);
    }

    return color;
}