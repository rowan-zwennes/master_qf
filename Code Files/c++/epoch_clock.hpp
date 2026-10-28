// Funding epoch clock: time remaining to the next settlement.

#pragma once

#include <cstdint>

namespace hft {

class EpochClock {
public:
    explicit EpochClock(std::int64_t epoch_period_ms = 8 * 3600 * 1000,
                        std::int64_t reference_settlement_ms = 0);

    std::int64_t current_epoch_start_ms(std::int64_t t_ms) const;
    std::int64_t next_settlement_ms(std::int64_t t_ms) const;
    double       seconds_to_funding(std::int64_t t_ms) const;
    bool         in_boundary_layer(std::int64_t t_ms, double u_star_s) const;
    bool         is_settlement_tick(std::int64_t t_ms) const;

private:
    std::int64_t epoch_period_ms_{};
    std::int64_t reference_settlement_ms_{};
};

} // namespace hft
