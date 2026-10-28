// Exponentially weighted moving-average drift estimator.

#pragma once

#include <cmath>
#include <cstdint>

namespace hft {

class EmaDrift {
public:
    explicit EmaDrift(double half_life_seconds = 10.0) noexcept
        : half_life_s_(half_life_seconds) {}

    void reset() noexcept {
        alpha_ = 0.0;
        last_price_ = 0.0;
        last_ms_ = -1;
    }

    double update(std::int64_t ts_ms, double price) noexcept {
        if (last_ms_ >= 0 && ts_ms > last_ms_) {
            const double dt = static_cast<double>(ts_ms - last_ms_) / 1000.0;
            const double slope = (price - last_price_) / dt;
            const double w = std::exp(-kLn2 * dt / half_life_s_);
            alpha_ = w * alpha_ + (1.0 - w) * slope;
        }
        last_price_ = price;
        last_ms_ = ts_ms;
        return alpha_;
    }

    double alpha() const noexcept { return alpha_; }

private:
    static constexpr double kLn2 = 0.6931471805599453;
    double half_life_s_{10.0};
    double alpha_{0.0};
    double last_price_{0.0};
    std::int64_t last_ms_{-1};
};

} // namespace hft
