// Fonction de calcul de la couleur du ciel atmosphérique
float3 calculateProceduralSky(float3 rayDir, float3 sunDir, float3 sunColor) {
    float3 v = normalize(rayDir);
    float3 s = normalize(sunDir);

    float cosTheta = dot(v, s);
    float height = max(v.y, 0.0); // Hauteur dans le ciel (0 a l'horizon, 1 au zenith)

    float sunElevation = s.y;

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

    float3 finalSky = (skyBase + mieColor + sunColor) * horizonFade;

    if (sunElevation < 0.0) {
        float nightFactor = saturate(-sunElevation * 3.0);
        finalSky = lerp(finalSky, float3(0.002, 0.004, 0.01), nightFactor);
    }

    return finalSky;
}