// Top-of-book state on the replay grid.

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace hft {

struct Level { double price; double qty; };

class OrderBook {
public:
    static constexpr std::size_t kDepth = 20;

    // Full top-L snapshot (level 0 = touch). n may be < kDepth.
    void apply_snapshot(const double* bid_p, const double* bid_q,
                        const double* ask_p, const double* ask_q,
                        std::size_t n, std::int64_t ts_ms) noexcept {
        n_ = n < kDepth ? n : kDepth;
        for (std::size_t i = 0; i < n_; ++i) {
            bids_[i] = {bid_p[i], bid_q[i]};
            asks_[i] = {ask_p[i], ask_q[i]};
        }
        last_event_time_ms_ = ts_ms;
    }

    double best_bid() const noexcept { return bids_[0].price; }
    double best_ask() const noexcept { return asks_[0].price; }
    double mid() const noexcept {
        return 0.5 * (bids_[0].price + asks_[0].price);
    }
    double micro_price() const noexcept {
        const double bq = bids_[0].qty, aq = asks_[0].qty;
        const double den = bq + aq;
        return den > 1e-12
            ? (bids_[0].price * aq + asks_[0].price * bq) / den
            : mid();
    }
    double spread() const noexcept {
        return asks_[0].price - bids_[0].price;
    }
    bool sane() const noexcept {
        return n_ > 0 && bids_[0].price > 0.0
            && asks_[0].price > bids_[0].price;
    }
    const Level& bid(std::size_t i) const noexcept { return bids_[i]; }
    const Level& ask(std::size_t i) const noexcept { return asks_[i]; }
    std::size_t depth() const noexcept { return n_; }
    std::int64_t last_event_time_ms() const noexcept {
        return last_event_time_ms_;
    }

private:
    std::array<Level, kDepth> bids_{};
    std::array<Level, kDepth> asks_{};
    std::size_t n_{0};
    std::int64_t last_event_time_ms_{0};
};

} // namespace hft
