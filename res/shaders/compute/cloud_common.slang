#define CLOUD_COVERAGE 0.60

float hash31(float3 p) {
    p = frac(p * float3(443.897, 441.423, 437.195));
    p += dot(p, p.yzx + 19.19);
    return frac((p.x + p.y) * p.z);
}

float noise3D(float3 p) {
    float3 i = floor(p);
    float3 f = frac(p);
    float3 u = f * f * (3.0 - 2.0 * f);

    float n000 = hash31(i + float3(0.0, 0.0, 0.0));
    float n100 = hash31(i + float3(1.0, 0.0, 0.0));
    float n010 = hash31(i + float3(0.0, 1.0, 0.0));
    float n110 = hash31(i + float3(1.0, 1.0, 0.0));
    float n001 = hash31(i + float3(0.0, 0.0, 1.0));
    float n101 = hash31(i + float3(1.0, 0.0, 1.0));
    float n011 = hash31(i + float3(0.0, 1.0, 1.0));
    float n111 = hash31(i + float3(1.0, 1.0, 1.0));

    float nx00 = lerp(n000, n100, u.x);
    float nx10 = lerp(n010, n110, u.x);
    float nx01 = lerp(n001, n101, u.x);
    float nx11 = lerp(n011, n111, u.x);

    float nxy0 = lerp(nx00, nx10, u.y);
    float nxy1 = lerp(nx01, nx11, u.y);

    return lerp(nxy0, nxy1, u.z);
}

float fbm3D(float3 p) {
    float val = 0.0;
    float amp = 0.5;
    float freq = 1.0;
    [unroll]
    for (int i = 0; i < 4; ++i) { 
        val += amp * noise3D(p * freq);
        freq *= 2.02;
        amp *= 0.5;
    }
    return val;
}

float sampleCloudDensity(float3 pos, float time) {
    float H_min = 20.0;
    float H_max = 35.0;

    if (pos.y < H_min || pos.y > H_max) return 0.0;

    // Hauteur normalisée dans la couche nuageuse [0, 1]
    float h = (pos.y - H_min) / (H_max - H_min);

    // Profil de hauteur d'un Cumulus : base plate (h=0) et sommets arrondis (h=1)
    float heightGradient = smoothstep(0.0, 0.15, h) * smoothstep(1.0, 0.65, h);

    // Vent 3D (déplacement des coordonnées dans le temps)
    float3 wind = float3(time * 0.03, 0.0, time * 0.015);
    float3 uvw = pos * 0.08 + wind;

    // Échantillonnage FBM 3D
    float n = fbm3D(uvw);

    // Couverture du nuage
    float coverage = CLOUD_COVERAGE;
    float density = smoothstep(coverage, coverage + 0.35, n) * heightGradient;

    return density;
}