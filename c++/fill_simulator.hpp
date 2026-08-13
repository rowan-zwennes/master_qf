// Last-in-queue fill matching against the trade stream.

#pragma once

#include <cstdint>
#include <vector>

#include "queue_position.hpp"

namespace hft {

class FillSimulator {
public:
    // Place a resting maker quote; returns its stable id.
    int place(int side, double price, double our_qty, double queue_ahead,
              std::uint64_t ts_ms) {
        return tracker_.place(side, price, our_qty, queue_ahead, ts_ms);
    }

    // Drive one trade print; returns the fills it produced.
    std::vector<FillEvent> on_trade(double price, double qty, int taker_side,
                                    std::uint64_t ts_ms) {
        return tracker_.on_trade(price, qty, taker_side, ts_ms);
    }

    void cancel(int id) { tracker_.cancel(id); }
    void cancel_all() { tracker_.cancel_all(); }
    std::size_t active_count() const { return tracker_.active_count(); }
    const PassiveOrder* find(int id) const { return tracker_.find(id); }

    QueuePositionTracker& tracker() { return tracker_; }

private:
    QueuePositionTracker tracker_;
};

} // namespace hft
