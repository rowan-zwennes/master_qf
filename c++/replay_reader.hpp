// Reader for the binary replay input format.

#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace hft {

struct ReplayDay {
    // header
    std::uint32_t n_ticks{}, n_trades{}, n_levels{}, n_settle{}, n_pause{};
    std::int64_t t0_ms{};
    double tick{}, quote_size{};
    // book, 100 ms grid (row-major, level 0 = touch)
    std::vector<std::int64_t> tick_ts;
    std::vector<std::uint8_t> valid;
    std::vector<double> bid_p, ask_p;
    std::vector<float> bid_q, ask_q;
    std::vector<double> mid, micro, alpha, alpha_ar, sigma, A, k, f_rate, u_s;
    // trades, true event clock
    std::vector<std::int64_t> trade_ts;
    std::vector<double> trade_px, trade_qty;
    std::vector<std::int8_t> trade_side;
    // settlement calendar with (mark, rate) resolved at export time
    std::vector<std::int64_t> settle_ts;
    std::vector<double> settle_mark, settle_rate;
    // hard-pause intervals (start, end) ms, half-open
    std::vector<std::int64_t> pause;
};

class ReplayReader {
public:
    // Parse + validate; false with a message in last_error() on failure.
    bool open(std::string_view path);
    const ReplayDay& day() const noexcept { return day_; }
    const std::string& last_error() const noexcept { return err_; }

private:
    ReplayDay day_{};
    std::string err_;
};

} // namespace hft
