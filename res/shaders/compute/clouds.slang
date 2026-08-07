import cloud_common;

#define CLOUD_QUALITY 1

float4 calculateClouds(float3 rayDir, float3 sunDir, float3 sunColor, float time, float nightFactor, int mipLevel) {
    if (rayDir.y <= 0.005) return float4(0.0);

    float H_min = 20.0;
    float H_max = 35.0;

    // Intersection de la couche nuageuse 3D
    float t_min = H_min / rayDir.y;
    float t_max = H_max / rayDir.y;

    // Limiter la longueur maximale du rayon vers l'horizon pour conserver les performances
    float maxDist = min(t_max - t_min, 70.0);
    t_max = t_min + maxDist;

    // Nombre d'échantillons le long du rayon
    int steps;
    if (mipLevel > 0) {
        #if CLOUD_QUALITY == 1
            steps = 0;
        #elif CLOUD_QUALITY == 2
            steps = 1;
        #elif CLOUD_QUALITY == 3
            steps = 2;
        #endif
    } else {
        #if CLOUD_QUALITY == 1
            steps = 4;
        #elif CLOUD_QUALITY == 2
            steps = 6;
        #elif CLOUD_QUALITY == 3
            steps = 10;
        #endif
    }
    float stepSize = (t_max - t_min) / float(steps);

    float3 accumulatedColor = float3(0.0);
    float transmittance = 1.0; // Opacité restante (Loi de Beer-Lambert)

    float cosTheta = dot(rayDir, sunDir);
    float forwardScattering = pow(max(cosTheta, 0.0), 8.0) * 2.2;

    float horizonFade = smoothstep(0.005, 0.12, rayDir.y);

    for (int i = 0; i < steps; ++i) {
        if (transmittance < 0.01) break; // Rayon totalement absorbé

        float t = t_min + (float(i) + 0.5) * stepSize;
        float3 p = rayDir * t;

        float density = sampleCloudDensity(p, time) * horizonFade;

        if (density > 0.01) {
            // --- AUTO-OMBRAGE (Self-Shadowing) ---
            // On sonde la densité du nuage vers le soleil pour calculer l'ombre portée interne
            float3 sunSamplePos = p + sunDir * 3.5;
            float sunDensity = density;
            float shadowTransmittance = exp(-sunDensity * 3.0); // Atténuation de la lumière du soleil

            // Lumière incidente au point P
            float3 daySun = sunColor * (shadowTransmittance * 1.3 + forwardScattering);
            float3 dayAmbient = float3(0.55, 0.68, 0.85) * 0.7;
            float3 dayLight = lerp(daySun, dayAmbient, 0.35);

            float3 nightLight = float3(0.015, 0.025, 0.05);
            float3 stepLight = lerp(dayLight, nightLight, nightFactor);

            // Intégration de la couleur et de l'opacité selon Beer-Lambert
            float opacity = 1.0 - exp(-density * stepSize * 0.2);
            accumulatedColor += stepLight * opacity * transmittance;
            transmittance *= (1.0 - opacity);
        }
    }

    return float4(accumulatedColor, (1.0 - transmittance));
}