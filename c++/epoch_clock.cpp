// Funding epoch clock.

#include "epoch_clock.hpp"

namespace hft {

EpochClock::EpochClock(std::int64_t epoch_period_ms,
                       std::int64_t reference_settlement_ms)
    : epoch_period_ms_(epoch_period_ms),
      reference_settlement_ms_(reference_settlement_ms) {}

std::int64_t EpochClock::current_epoch_start_ms(std::int64_t t_ms) const {
    // floor-div towards -inf so pre-reference times behave
    std::int64_t d = t_ms - reference_settlement_ms_;
    std::int64_t k = d / epoch_period_ms_;
    if (d % epoch_period_ms_ < 0) --k;
    return reference_settlement_ms_ + k * epoch_period_ms_;
}

std::int64_t EpochClock::next_settlement_ms(std::int64_t t_ms) const {
    return current_epoch_start_ms(t_ms) + epoch_period_ms_;
}

double EpochClock::seconds_to_funding(std::int64_t t_ms) const {
    return static_cast<double>(next_settlement_ms(t_ms) - t_ms) / 1000.0;
}

bool EpochClock::in_boundary_layer(std::int64_t t_ms, double u_star_s) const {
    return seconds_to_funding(t_ms) <= u_star_s;
}

bool EpochClock::is_settlement_tick(std::int64_t t_ms) const {
    return (t_ms - reference_settlement_ms_) % epoch_period_ms_ == 0;
}

} // namespace hft
