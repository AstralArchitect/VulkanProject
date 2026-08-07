#include <glm/glm.hpp>
#include <cmath>
#include <algorithm>

namespace SunCalc {

    constexpr double PI = 3.14159265359;
    constexpr double DEG_TO_RAD = 3.14159265358979323846 / 180.0;
    constexpr double RAD_TO_DEG = 180.0 / 3.14159265358979323846;

    // Convertit Date & Heure UTC en Jour Julien (JD)
    inline double calculateJulianDay(int year, int month, int day, double hourUTC) {
        if (month <= 2) {
            year -= 1;
            month += 12;
        }
        int A = year / 100;
        int B = 2 - A + (A / 4);
        double jd = std::floor(365.25 * (year + 4716)) 
                + std::floor(30.6001 * (month + 1)) 
                + day + B - 1524.5;
        return jd + (hourUTC / 24.0);
    }

    glm::vec3 calculateSunDirection(int year, int month, int day, double hourUTC, double latitudeDeg, double longitudeDeg);
}