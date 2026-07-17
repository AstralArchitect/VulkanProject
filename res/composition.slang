[vk::binding(0, 0)] Texture2D<float4> inDirectColor;
[vk::binding(1, 0)] Texture2D<float4> inReflection;
[vk::binding(2, 0)] Texture2D<float4> inNormal;
[vk::binding(3, 0)] Texture2D<float> inDepth;

[[vk::binding(4, 0)]] RWTexture2D<float4> outFinalImage;

static const int FILTER_RADIUS = 2;          // Rayon de boucle
static const int STEP_SIZE = 1;              // Espacement
static const float SIGMA_SPATIAL = 2.0;      // Douceur spatiale
static const float SIGMA_DEPTH = 0.01;       // Tolérance profondeur (ajustée pour être relative)
static const float SIGMA_NORMAL = 128.0;     // Sensibilité aux normales
static const float SIGMA_LUMINANCE = 0.5;    // Sensibilité à la luminance (préserve les détails brillants)

// Fonction utilitaire pour extraire la luminance d'une couleur
float getLuminance(float3 color) {
    return dot(color, float3(0.2126, 0.7152, 0.0722));
}

// Fonction de débruitage améliorée (Bilateral adaptatif)
float3 denoiseReflection(int2 texCoords, uint width, uint height) {
    float3 centerReflection = inReflection.Load(int3(texCoords, 0)).rgb;
    
    // Si le pixel n'a aucune réflexion (noir total), on gagne du temps (early-out)
    if (dot(centerReflection, centerReflection) == 0.0) {
        return centerReflection;
    }

    float3 centerNormal = inNormal.Load(int3(texCoords, 0)).xyz;
    float centerDepth = inDepth.Load(int3(texCoords, 0)).r;
    float centerLuma = getLuminance(centerReflection);

    float3 sumColor = float3(0.0);
    float sumWeight = 0.0;

    for (int y = -FILTER_RADIUS; y <= FILTER_RADIUS; ++y) {
        for (int x = -FILTER_RADIUS; x <= FILTER_RADIUS; ++x) {
            
            // Sparse sampling : on écarte les échantillons pour flouter plus large
            int2 offset = int2(x, y) * STEP_SIZE;
            int2 neighborCoords = texCoords + offset;

            // Clamping pour ne pas lire hors de l'écran
            neighborCoords = clamp(neighborCoords, int2(0, 0), int2(width - 1, height - 1));

            float3 neighborReflection = inReflection.Load(int3(neighborCoords, 0)).rgb;
            float3 neighborNormal = inNormal.Load(int3(neighborCoords, 0)).xyz;
            float neighborDepth = inDepth.Load(int3(neighborCoords, 0)).r;
            float neighborLuma = getLuminance(neighborReflection);

            // 1. Poids Spatial : plus c'est loin, moins ça pèse
            float spatialDist2 = float(x * x + y * y); // Attention, ici x et y sans le STEP pour un poids correct
            float spatialWeight = exp(-spatialDist2 / (2.0 * SIGMA_SPATIAL * SIGMA_SPATIAL));

            // 2. Poids Profondeur : Différence relative (plus robuste selon la distance de la caméra)
            float depthDiff = abs(centerDepth - neighborDepth) / max(centerDepth, 0.0001);
            float depthWeight = exp(-depthDiff / SIGMA_DEPTH);

            // 3. Poids Normales : Coupe le flou sur les arêtes géométriques
            float normalDot = max(dot(centerNormal, neighborNormal), 0.0);
            float normalWeight = pow(normalDot, SIGMA_NORMAL);

            // 4. Poids Luminance (NOUVEAU) : Coupe le flou si le reflet change brutalement (ex: bord d'un miroir)
            float lumaDiff = abs(centerLuma - neighborLuma);
            float lumaWeight = exp(-lumaDiff / SIGMA_LUMINANCE);

            // Poids final
            float weight = spatialWeight * depthWeight * normalWeight * lumaWeight;

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

    // Composition finale
    float3 color = directColor.rgb + filteredReflection;

    // Tonemapping de base (Reinhard) + Gamma correction
    color = color / (color + float3(1.0));
    color = pow(color, float3(1.0 / 2.2));

    outFinalImage[texCoords] = float4(color, 1.0);
}