// Funding drain term applied to the reservation price.

#pragma once

#include <cmath>

namespace hft {

inline double F_funding(double s_ref, double f_t) noexcept {
    return s_ref * f_t;
}

// u_seconds_to_funding must be >= 0 (clamp upstream; no branch here).
inline double phi_funding(int q, double k_intensity, double rho, double F_t,
                          double u_seconds_to_funding) noexcept {
    return k_intensity * static_cast<double>(q) * F_t
           * std::exp(-rho * u_seconds_to_funding);
}

} // namespace hft
