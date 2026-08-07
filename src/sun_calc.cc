#include "sun_calc.hh"

glm::vec3 SunCalc::calculateSunDirection(int year, int month, int day, double hourUTC, double latitudeDeg, double longitudeDeg) {
    // 1. Nombre de jours depuis l'époque J2000.0 (1er jan 2000 à 12h UTC)
    double jd = calculateJulianDay(year, month, day, hourUTC);
    double d = jd - 2451545.0;

    // 2. Longitude moyenne (L) et Anomalie moyenne (g) en degrés
    double L = std::fmod(280.460 + 0.9856474 * d, 360.0);
    if (L < 0) L += 360.0;

    double g = std::fmod(357.528 + 0.9856003 * d, 360.0);
    if (g < 0) g += 360.0;
    double gRad = g * DEG_TO_RAD;

    // Longitude écliptique du soleil (lambda)
    double lambda = L + 1.915 * std::sin(gRad) + 0.020 * std::sin(2.0 * gRad);
    double lambdaRad = lambda * DEG_TO_RAD;

    // Obliquité de l'écliptique (epsilon)
    double epsilon = (23.439 - 0.0000004 * d) * DEG_TO_RAD;

    // 3. Coordonnées équatoriales (Ascension Droite alpha, Déclinaison delta)
    double alpha = std::atan2(std::cos(epsilon) * std::sin(lambdaRad), std::cos(lambdaRad));
    double delta = std::asin(std::sin(epsilon) * std::sin(lambdaRad));

    // 4. Temps sidéral local et Angle Horaire (H)
    double gmst = std::fmod(280.46061837 + 360.98564736629 * d, 360.0);
    if (gmst < 0) gmst += 360.0;

    double lst = gmst + longitudeDeg; // Temps sidéral local en degrés
    double H = (lst * DEG_TO_RAD) - alpha;

    // 5. Passage aux coordonnées horizontales (Hauteur alt et Azimut az)
    double latRad = latitudeDeg * DEG_TO_RAD;
    double sinAlt = std::sin(latRad) * std::sin(delta) + std::cos(latRad) * std::cos(delta) * std::cos(H);
    double alt = std::asin(sinAlt); // Hauteur en radians (-PI/2 sous la terre, +PI/2 au zénith)

    // Azimut (0° = Nord, 90° = Est, 180° = Sud, 270° = Ouest)
    double cosAz = (std::sin(delta) - std::sin(latRad) * std::sin(alt)) / (std::cos(latRad) * std::cos(alt));
    cosAz = std::clamp(cosAz, -1.0, 1.0);
    double az = std::acos(cosAz);
    if (std::sin(H) > 0) {
        az = 2.0 * PI - az;
    }

    // 6. Conversion en vecteur directionnel 3D
    // Repère par défaut : +Y = Haut, +Z = Nord, +X = Est
    float x = static_cast<float>(std::cos(alt) * std::sin(az));
    float y = static_cast<float>(std::sin(alt));
    float z = static_cast<float>(std::cos(alt) * std::cos(az));

    return glm::normalize(glm::vec3(x, y, z));
}