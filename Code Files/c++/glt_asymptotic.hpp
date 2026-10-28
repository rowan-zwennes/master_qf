// Closed-form Gueant-Lehalle-Tapia asymptotic quotes (Regime I).

#pragma once

#include <cmath>

namespace hft::glt {

struct MarketConsts {
    double gamma{};
    double sigma{};
    double A{};
    double k{};
    double c1{};        // (1/γ) ln(1 + γ/k)
    double scale{};     // sqrt(σ²γ/(2kA) (1+γ/k)^{1+k/γ})
    double inv_gs2{};   // 1 / (γσ²)

    MarketConsts() = default;
    MarketConsts(double g, double s, double A_, double k_) noexcept
        : gamma(g), sigma(s), A(A_), k(k_) {
        c1 = (1.0 / g) * std::log1p(g / k_);
        scale = std::sqrt(s * s * g / (2.0 * k_ * A_)
                          * std::pow(1.0 + g / k_, 1.0 + k_ / g));
        inv_gs2 = 1.0 / (g * s * s);
    }
};

struct Depths {
    double bid;   // δ_b from the anchor (mid)
    double ask;   // δ_a from the anchor
};

inline Depths depths(const MarketConsts& mc, int q, double alpha) noexcept {
    const double shift = alpha * mc.inv_gs2;
    return {mc.c1 + (-shift + q + 0.5) * mc.scale,
            mc.c1 + (shift - q + 0.5) * mc.scale};
}

inline double ml_quote_shift(const MarketConsts& mc, double alpha) noexcept {
    return alpha * mc.inv_gs2 * mc.scale;
}

} // namespace hft::glt
